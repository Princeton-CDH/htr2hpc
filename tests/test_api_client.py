"""Tests for htr2hpc.api_client — API client utilities and data structures."""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from htr2hpc.api_client import (
    RESULTCLASS_REGISTRY,
    OCRModel,
    ResultsList,
    Task,
    eScriptoriumAPIClient,
    to_namedtuple,
)

# Names pre-registered in RESULTCLASS_REGISTRY at module load time
_BUILTIN_REGISTRY_KEYS = set(RESULTCLASS_REGISTRY.keys())


@pytest.fixture(autouse=True)
def clean_registry():
    """Remove any dynamically-added entries from RESULTCLASS_REGISTRY after each test."""
    yield
    for key in list(RESULTCLASS_REGISTRY.keys()):
        if key not in _BUILTIN_REGISTRY_KEYS:
            del RESULTCLASS_REGISTRY[key]


@pytest.fixture
def api_client_instance():
    """An eScriptoriumAPIClient pointed at a fake base URL."""
    return eScriptoriumAPIClient(
        base_url="https://escriptorium.example.com",
        api_token="test-token-abc123",
    )


# ---------------------------------------------------------------------------
# to_namedtuple
# ---------------------------------------------------------------------------


def test_to_namedtuple_dict_creates_namedtuple():
    result = to_namedtuple("fruit", {"name": "apple", "color": "red"})
    assert result.name == "apple"
    assert result.color == "red"


def test_to_namedtuple_list_converts_each_item():
    result = to_namedtuple("tag", [{"id": 1}, {"id": 2}])
    assert len(result) == 2
    assert result[0].id == 1
    assert result[1].id == 2


def test_to_namedtuple_scalar_returned_unchanged():
    assert to_namedtuple("x", 42) == 42
    assert to_namedtuple("x", "hello") == "hello"
    assert to_namedtuple("x", None) is None


def test_to_namedtuple_nested_dict_converted_recursively():
    result = to_namedtuple("doc", {"id": 1, "owner": {"username": "alice"}})
    assert result.id == 1
    assert result.owner.username == "alice"


def test_to_namedtuple_nested_list_converted_recursively():
    result = to_namedtuple("doc", {"id": 1, "tags": [{"name": "a"}, {"name": "b"}]})
    assert result.tags[0].name == "a"
    assert result.tags[1].name == "b"


def test_to_namedtuple_reuses_registered_class():
    # OCRModel is pre-registered; to_namedtuple("model", ...) should use it
    data = {
        "pk": 1,
        "name": "test",
        "file": "/models/test.mlmodel",
        "file_size": 1000,
        "job": "Recognize",
        "owner": "alice",
        "training": False,
        "versions": [],
        "documents": [],
        "accuracy_percent": 0.95,
        "training_accuracy": 0.95,
        "rights": "private",
        "can_share": True,
    }
    result = to_namedtuple("model", data)
    assert isinstance(result, OCRModel)
    assert result.name == "test"


def test_to_namedtuple_caches_new_class():
    name = "_test_cache_check"
    to_namedtuple(name, {"a": 1})
    assert name in RESULTCLASS_REGISTRY


def test_to_namedtuple_plural_key_strips_s_for_nested_type():
    # Keys ending in 's' produce singular namedtuple type names for nested items
    result = to_namedtuple("doc", {"items": [{"id": 1}]})
    assert result.items[0].id == 1


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

TASK_DATA = {
    "pk": 7,
    "document": 3,
    "document_part": 12,
    "workflow_state": 3,
    "label": "export",
    "messages": "",
    "queued_at": "2024-01-15T10:00:00",
    "started_at": "2024-01-15T10:00:05",
    "done_at": "2024-01-15T10:01:05",
    "method": "export",
    "user": 1,
}


def test_task_post_init_converts_dates():
    task = Task(**TASK_DATA)
    assert isinstance(task.queued_at, datetime.datetime)
    assert isinstance(task.started_at, datetime.datetime)
    assert isinstance(task.done_at, datetime.datetime)


def test_task_duration_returns_timedelta():
    task = Task(**TASK_DATA)
    assert task.duration() == datetime.timedelta(minutes=1)


def test_task_duration_none_when_not_done():
    data = dict(TASK_DATA, done_at=None)
    task = Task(**data)
    assert task.duration() is None


def test_task_post_init_handles_none_dates():
    data = dict(TASK_DATA, started_at=None, done_at=None)
    task = Task(**data)
    assert task.started_at is None
    assert task.done_at is None


def test_task_duration_when_started_at_none():
    # done_at is set but started_at is None — subtraction would fail;
    # duration should still return a value (datetime - None raises TypeError)
    # In the current implementation, duration only checks done_at.
    # This test documents that behavior.
    data = dict(TASK_DATA, started_at=None)
    task = Task(**data)
    with pytest.raises(TypeError):
        task.duration()


# ---------------------------------------------------------------------------
# eScriptoriumAPIClient — construction
# ---------------------------------------------------------------------------


def test_client_strips_trailing_slash():
    client = eScriptoriumAPIClient(
        base_url="https://escriptorium.example.com/",
        api_token="tok",
    )
    assert client.base_url == "https://escriptorium.example.com"
    assert client.api_root == "https://escriptorium.example.com/api"


def test_client_sets_auth_header():
    client = eScriptoriumAPIClient(
        base_url="https://escriptorium.example.com",
        api_token="mytoken",
    )
    assert client.session.headers["Authorization"] == "Token mytoken"


def test_client_sets_user_agent(api_client_instance):
    assert "htr2hpc/" in api_client_instance.session.headers["User-Agent"]


# ---------------------------------------------------------------------------
# eScriptoriumAPIClient.export_file_url
# ---------------------------------------------------------------------------


def test_export_file_url_basic(api_client_instance):
    dt = datetime.datetime(2024, 3, 15, 9, 30)
    url = api_client_instance.export_file_url(
        user_id=5,
        document_id=42,
        document_name="My Document",
        file_format="alto",
        creation_time=dt,
    )
    assert url.startswith("https://escriptorium.example.com/media/users/5/")
    assert "doc42" in url
    assert "alto" in url
    assert "202403150930" in url


def test_export_file_url_slugifies_document_name(api_client_instance):
    dt = datetime.datetime(2024, 1, 1, 0, 0)
    url = api_client_instance.export_file_url(
        user_id=1,
        document_id=1,
        document_name="Héros & Dragons — 2024",
        file_format="alto",
        creation_time=dt,
    )
    # slugify turns special chars into ASCII; hyphens become underscores
    assert "/media/users/1/" in url
    # should not contain raw special characters
    assert "&" not in url
    assert "—" not in url


def test_export_file_url_truncates_long_name(api_client_instance):
    dt = datetime.datetime(2024, 1, 1, 0, 0)
    long_name = "A" * 100
    url = api_client_instance.export_file_url(
        user_id=1,
        document_id=1,
        document_name=long_name,
        file_format="alto",
        creation_time=dt,
    )
    filename = url.split("/")[-1]
    # slugified name is truncated to 32 chars in the implementation
    assert len(filename) < 120  # sanity check it didn't blow up


def test_export_file_url_ends_with_zip(api_client_instance):
    dt = datetime.datetime(2024, 6, 1, 12, 0)
    url = api_client_instance.export_file_url(
        user_id=2,
        document_id=10,
        document_name="test",
        file_format="alto",
        creation_time=dt,
    )
    assert url.endswith(".zip")


# ---------------------------------------------------------------------------
# eScriptoriumAPIClient._make_request — error handling
# ---------------------------------------------------------------------------


def test_make_request_raises_not_found(api_client_instance):
    from htr2hpc.api_client import NotFound

    with (
        patch.object(
            api_client_instance.session,
            "get",
            return_value=MagicMock(status_code=404),
        ),
        pytest.raises(NotFound),
    ):
        api_client_instance._make_request("documents/999/")


def test_make_request_raises_not_allowed_on_401(api_client_instance):
    from htr2hpc.api_client import NotAllowed

    with (
        patch.object(
            api_client_instance.session,
            "get",
            return_value=MagicMock(status_code=401),
        ),
        pytest.raises(NotAllowed),
    ):
        api_client_instance._make_request("documents/1/")


def test_make_request_raises_not_allowed_on_403(api_client_instance):
    from htr2hpc.api_client import NotAllowed

    with (
        patch.object(
            api_client_instance.session,
            "get",
            return_value=MagicMock(status_code=403),
        ),
        pytest.raises(NotAllowed),
    ):
        api_client_instance._make_request("documents/1/")


def test_make_request_raises_on_unsupported_method(api_client_instance):
    with pytest.raises(ValueError, match="unsupported http method"):
        api_client_instance._make_request("documents/1/", method="PATCH")


def test_make_request_uses_absolute_url_as_is(api_client_instance):
    mock_resp = MagicMock(status_code=200)
    with patch.object(
        api_client_instance.session, "get", return_value=mock_resp
    ) as mock_get:
        api_client_instance._make_request(
            "https://escriptorium.example.com/api/documents/1/"
        )
        called_url = mock_get.call_args[0][0]
        assert called_url == "https://escriptorium.example.com/api/documents/1/"


def test_make_request_prepends_api_root_for_relative_url(api_client_instance):
    mock_resp = MagicMock(status_code=200)
    with patch.object(
        api_client_instance.session, "get", return_value=mock_resp
    ) as mock_get:
        api_client_instance._make_request("documents/1/")
        called_url = mock_get.call_args[0][0]
        assert called_url == "https://escriptorium.example.com/api/documents/1/"


# ---------------------------------------------------------------------------
# model_create — validation
# ---------------------------------------------------------------------------


def test_model_create_raises_on_invalid_job(api_client_instance, tmp_path):
    fake_model = tmp_path / "model.mlmodel"
    fake_model.write_bytes(b"fake")
    with pytest.raises(ValueError, match="not a valid model job name"):
        api_client_instance.model_create(fake_model, job="Train")


# ---------------------------------------------------------------------------
# eScriptoriumAPIClient._make_request — success + method routing
# ---------------------------------------------------------------------------


def test_make_request_returns_response_on_200(api_client_instance):
    mock_resp = MagicMock(status_code=200)
    with patch.object(api_client_instance.session, "get", return_value=mock_resp):
        result = api_client_instance._make_request("documents/1/")
    assert result is mock_resp


def test_make_request_post_routes_to_session_post(api_client_instance):
    mock_resp = MagicMock(status_code=200)
    with patch.object(
        api_client_instance.session, "post", return_value=mock_resp
    ) as mock_post:
        api_client_instance._make_request("documents/", method="POST", data={"k": "v"})
    mock_post.assert_called_once()


def test_make_request_put_routes_to_session_put(api_client_instance):
    mock_resp = MagicMock(status_code=200)
    with patch.object(
        api_client_instance.session, "put", return_value=mock_resp
    ) as mock_put:
        api_client_instance._make_request("documents/1/", method="PUT", data={"k": "v"})
    mock_put.assert_called_once()


def test_make_request_delete_routes_to_session_delete(api_client_instance):
    mock_resp = MagicMock(status_code=204)
    with patch.object(
        api_client_instance.session, "delete", return_value=mock_resp
    ) as mock_del:
        api_client_instance._make_request(
            "documents/1/", method="DELETE", expected_status=204
        )
    mock_del.assert_called_once()


# ---------------------------------------------------------------------------
# get_model_accuracy
# ---------------------------------------------------------------------------


def test_get_model_accuracy_extracts_last_value(tmp_path):
    import json

    from htr2hpc.api_client import get_model_accuracy

    fake_model = tmp_path / "model.mlmodel"
    fake_model.write_bytes(b"fake")

    mock_spec = MagicMock()
    mock_spec.description.metadata.userDefined = {
        "kraken_meta": json.dumps({"accuracy": [[0, 0.80], [1, 0.90], [2, 0.95]]})
    }
    mock_ml_model = MagicMock()
    mock_ml_model.get_spec.return_value = mock_spec

    with patch(
        "htr2hpc.api_client.coremltools.models.MLModel", return_value=mock_ml_model
    ):
        result = get_model_accuracy(fake_model)
    assert result == 0.95


# ---------------------------------------------------------------------------
# ResultsList.next_page
# ---------------------------------------------------------------------------


def test_results_list_next_page_returns_new_results_list(api_client_instance):

    next_url = "https://escriptorium.example.com/api/models/?page=2"
    page1 = ResultsList(
        api=api_client_instance,
        result_type="model",
        count=2,
        next=next_url,
        previous=None,
        results=[],
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "count": 2,
        "next": None,
        "previous": next_url,
        "results": [],
    }
    with patch.object(api_client_instance, "_make_request", return_value=mock_resp):
        page2 = page1.next_page()
    assert isinstance(page2, ResultsList)
    assert page2.next is None


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


def test_get_current_user_returns_namedtuple(api_client_instance):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"pk": 1, "username": "alice"}
    with patch.object(api_client_instance, "_make_request", return_value=mock_resp):
        user = api_client_instance.get_current_user()
    assert user.pk == 1
    assert user.username == "alice"


# ---------------------------------------------------------------------------
# model_list / model_details / model_delete
# ---------------------------------------------------------------------------


def test_model_list_no_page_returns_results_list(api_client_instance):

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "count": 0,
        "next": None,
        "previous": None,
        "results": [],
    }
    with patch.object(
        api_client_instance, "_make_request", return_value=mock_resp
    ) as mock_req:
        result = api_client_instance.model_list()
    assert isinstance(result, ResultsList)
    # no page param passed
    mock_req.assert_called_once_with("models/", params=None)


def test_model_list_with_page_passes_param(api_client_instance):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "count": 0,
        "next": None,
        "previous": None,
        "results": [],
    }
    with patch.object(
        api_client_instance, "_make_request", return_value=mock_resp
    ) as mock_req:
        api_client_instance.model_list(page=2)
    mock_req.assert_called_once_with("models/", params={"page": 2})


def test_model_details_returns_namedtuple(api_client_instance):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "pk": 7,
        "name": "latin",
        "file": "/models/latin.mlmodel",
        "file_size": 500,
        "job": "Recognize",
        "owner": "alice",
        "training": False,
        "versions": [],
        "documents": [],
        "accuracy_percent": 0.92,
        "training_accuracy": 0.92,
        "rights": "private",
        "can_share": False,
    }
    with patch.object(api_client_instance, "_make_request", return_value=mock_resp):
        model = api_client_instance.model_details(7)
    assert model.pk == 7
    assert model.name == "latin"


def test_model_delete_uses_delete_method(api_client_instance):
    with patch.object(api_client_instance, "_make_request") as mock_req:
        api_client_instance.model_delete(7)
    mock_req.assert_called_once_with("models/7/", method="DELETE", expected_status=204)


# ---------------------------------------------------------------------------
# model_update / model_create
# ---------------------------------------------------------------------------


def test_model_update_uses_existing_job_and_name_when_not_provided(
    api_client_instance, tmp_path
):
    fake_file = tmp_path / "updated.mlmodel"
    fake_file.write_bytes(b"fake model data")

    mock_details = MagicMock()
    mock_details.job = "Recognize"
    mock_details.name = "existing-name"

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "pk": 5,
        "name": "existing-name",
        "file": "/models/updated.mlmodel",
        "file_size": 15,
        "job": "Recognize",
        "owner": "alice",
        "training": False,
        "versions": [],
        "documents": [],
        "accuracy_percent": None,
        "training_accuracy": None,
        "rights": "private",
        "can_share": False,
    }
    with (
        patch.object(api_client_instance, "model_details", return_value=mock_details),
        patch.object(
            api_client_instance, "_make_request", return_value=mock_resp
        ) as mock_req,
        patch("htr2hpc.api_client.get_model_accuracy", return_value=0.91) as mock_acc,
    ):
        result = api_client_instance.model_update(5, fake_file)
    assert result.name == "existing-name"
    _, kwargs = mock_req.call_args
    assert kwargs["method"] == "PUT"
    assert kwargs["data"]["name"] == "existing-name"
    assert kwargs["data"]["job"] == "Recognize"
    assert kwargs["data"]["training_accuracy"] == 0.91
    mock_acc.assert_called_once_with(fake_file)


def test_model_create_uses_filename_stem_as_default_name(api_client_instance, tmp_path):
    fake_file = tmp_path / "foo.mlmodel"
    fake_file.write_bytes(b"fake model data")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "pk": 9,
        "name": "foo",
        "file": "/models/foo.mlmodel",
        "file_size": 15,
        "job": "Recognize",
        "owner": "alice",
        "training": False,
        "versions": [],
        "documents": [],
        "accuracy_percent": None,
        "training_accuracy": None,
        "rights": "private",
        "can_share": False,
    }
    with (
        patch.object(
            api_client_instance, "_make_request", return_value=mock_resp
        ) as mock_req,
        patch("htr2hpc.api_client.get_model_accuracy", return_value=0.88) as mock_acc,
    ):
        result = api_client_instance.model_create(fake_file, job="Recognize")
    assert result.name == "foo"
    _, kwargs = mock_req.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["expected_status"] == 201
    assert kwargs["data"]["name"] == "foo"
    assert kwargs["data"]["training_accuracy"] == 0.88
    mock_acc.assert_called_once_with(fake_file)


# ---------------------------------------------------------------------------
# document_list / document_details / document_parts_list
# ---------------------------------------------------------------------------


def test_document_list_returns_results_list(api_client_instance):

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "count": 0,
        "next": None,
        "previous": None,
        "results": [],
    }
    with patch.object(api_client_instance, "_make_request", return_value=mock_resp):
        result = api_client_instance.document_list()
    assert isinstance(result, ResultsList)


def test_document_list_passes_page_param(api_client_instance):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "count": 0,
        "next": None,
        "previous": None,
        "results": [],
    }
    with patch.object(
        api_client_instance, "_make_request", return_value=mock_resp
    ) as mock_req:
        api_client_instance.document_list(page=3)
    mock_req.assert_called_once_with("documents/", params={"page": 3})


def test_document_details_returns_namedtuple(api_client_instance):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"pk": 42, "name": "My Doc", "parts_count": 5}
    with patch.object(api_client_instance, "_make_request", return_value=mock_resp):
        doc = api_client_instance.document_details(42)
    assert doc.pk == 42
    assert doc.name == "My Doc"


def test_document_parts_list_returns_results_list(api_client_instance):

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "count": 0,
        "next": None,
        "previous": None,
        "results": [],
    }
    with patch.object(api_client_instance, "_make_request", return_value=mock_resp):
        result = api_client_instance.document_parts_list(42)
    assert isinstance(result, ResultsList)
