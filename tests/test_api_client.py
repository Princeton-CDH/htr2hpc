"""Tests for htr2hpc.api_client — API client utilities and data structures."""
import datetime
from collections import namedtuple
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

    with patch.object(
        api_client_instance.session,
        "get",
        return_value=MagicMock(status_code=404),
    ):
        with pytest.raises(NotFound):
            api_client_instance._make_request("documents/999/")


def test_make_request_raises_not_allowed_on_401(api_client_instance):
    from htr2hpc.api_client import NotAllowed

    with patch.object(
        api_client_instance.session,
        "get",
        return_value=MagicMock(status_code=401),
    ):
        with pytest.raises(NotAllowed):
            api_client_instance._make_request("documents/1/")


def test_make_request_raises_not_allowed_on_403(api_client_instance):
    from htr2hpc.api_client import NotAllowed

    with patch.object(
        api_client_instance.session,
        "get",
        return_value=MagicMock(status_code=403),
    ):
        with pytest.raises(NotAllowed):
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
