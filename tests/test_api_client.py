"""Tests for htr2hpc.api_client — API client utilities and data structures."""

import datetime
import json
import logging
from email import message_from_bytes, policy
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
import requests
import responses

from htr2hpc.api_client import (
    RESULTCLASS_REGISTRY,
    NotAllowed,
    NotFound,
    OCRModel,
    ResultsList,
    Task,
    eScriptoriumAPIClient,
    get_model_accuracy,
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


API_ROOT = "https://escriptorium.example.com/api"
EMPTY_PAGE = {"count": 0, "next": None, "previous": None, "results": []}
# a model as returned by the eScriptorium API
MODEL_JSON = {
    "pk": 1,
    "name": "model",
    "file": "/media/models/model.mlmodel",
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


@pytest.fixture
def rsps():
    """Fake eScriptorium server, built on `responses`, so request building and
    response handling run for real and we can assert on the actual HTTP traffic.
    Fails the test if a registered response is never requested, so a stale or
    misspelled url can't go unnoticed."""
    with responses.RequestsMock() as mock:
        yield mock


def sent_query(call):
    """Query string parameters of a request recorded by `responses`."""
    # keep blank values, so a stray empty parameter is not hidden from assertions
    return parse_qs(urlparse(call.request.url).query, keep_blank_values=True)


def sent_form(call):
    """Form-encoded body of a request recorded by `responses`."""
    # keep blank values, so a stray empty field is not hidden from assertions
    return parse_qs(call.request.body, keep_blank_values=True)


def sent_multipart(call):
    """Fields of a multipart/form-data request recorded by `responses`,
    as a dict of field name -> bytes."""
    # The body of a multipart request is a MIME message, so parse it with the
    # standard library's email parser. This lets tests assert on each field by
    # name, instead of searching for strings in the raw bytes.
    header = f"Content-Type: {call.request.headers['Content-Type']}\r\n\r\n"
    message = message_from_bytes(
        header.encode() + call.request.body, policy=policy.default
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(
            decode=True
        )
        for part in message.iter_parts()
    }


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
    dt = datetime.datetime(2024, 3, 15, 9, 30, tzinfo=datetime.UTC)
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
    dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.UTC)
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
    dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.UTC)
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
    dt = datetime.datetime(2024, 6, 1, 12, 0, tzinfo=datetime.UTC)
    url = api_client_instance.export_file_url(
        user_id=2,
        document_id=10,
        document_name="test",
        file_format="alto",
        creation_time=dt,
    )
    assert url.endswith(".zip")


# ---------------------------------------------------------------------------
# eScriptoriumAPIClient._make_request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "exception"),
    [(404, NotFound), (401, NotAllowed), (403, NotAllowed)],
    ids=["not-found", "unauthorized", "forbidden"],
)
def test_make_request_maps_error_status_to_exception(
    rsps, api_client_instance, status, exception
):
    rsps.get(f"{API_ROOT}/documents/1/", status=status)

    with pytest.raises(exception):
        api_client_instance._make_request("documents/1/")


def test_make_request_raises_on_unexpected_server_error(rsps, api_client_instance):
    rsps.get(f"{API_ROOT}/documents/", status=500)

    # anything other than the explicitly handled statuses
    # should surface as a requests HTTPError rather than be swallowed
    with pytest.raises(requests.HTTPError):
        api_client_instance._make_request("documents/")


def test_make_request_raises_on_unsupported_method(api_client_instance):
    with pytest.raises(ValueError, match="unsupported http method"):
        api_client_instance._make_request("documents/1/", method="PATCH")


@pytest.mark.parametrize(
    "url",
    ["documents/1/", f"{API_ROOT}/documents/1/"],
    ids=["relative", "absolute"],
)
def test_make_request_resolves_url_against_api_root(rsps, api_client_instance, url):
    rsps.get(f"{API_ROOT}/documents/1/")

    api_client_instance._make_request(url)

    # a relative url is added to the api root; an absolute url
    # (e.g. the "next" link of a paged result) is not prefixed a second time
    assert rsps.calls[0].request.url == f"{API_ROOT}/documents/1/"


def test_make_request_sends_params_data_and_files(rsps, api_client_instance):
    rsps.post(f"{API_ROOT}/models/", status=200)

    api_client_instance._make_request(
        "models/",
        method="POST",
        params={"page": 2},
        data={"name": "my model"},
        files={"file": ("m.mlmodel", b"model bytes")},
    )

    call = rsps.calls[0]
    assert sent_query(call) == {"page": ["2"]}
    # data and file are sent together as a multipart form
    assert sent_multipart(call) == {"name": b"my model", "file": b"model bytes"}


def test_make_request_delete_with_expected_status(rsps, api_client_instance):
    rsps.delete(f"{API_ROOT}/models/7/", status=204)

    response = api_client_instance._make_request(
        "models/7/", method="DELETE", expected_status=204
    )

    assert response.status_code == 204


# ---------------------------------------------------------------------------
# get_model_accuracy
# ---------------------------------------------------------------------------


def test_get_model_accuracy_extracts_last_value(mocker, tmp_path):
    # Kraken stores [epoch, accuracy] pairs in the model metadata; the last
    # one is the accuracy of the final training epoch. Loading a real model
    # needs a trained kraken model, so fake the coremltools model instead.
    fake_model = tmp_path / "model.mlmodel"
    fake_model.write_bytes(b"fake")

    mock_spec = MagicMock()
    mock_spec.description.metadata.userDefined = {
        "kraken_meta": json.dumps({"accuracy": [[0, 0.80], [1, 0.90], [2, 0.95]]})
    }
    mock_ml_model = MagicMock()
    mock_ml_model.get_spec.return_value = mock_spec
    mocker.patch(
        "htr2hpc.api_client.coremltools.models.MLModel", return_value=mock_ml_model
    )

    assert get_model_accuracy(fake_model) == 0.95


# ---------------------------------------------------------------------------
# single-item endpoints: one GET, result converted to a namedtuple
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "path", "payload", "field"),
    [
        (
            lambda api: api.get_current_user(),
            "users/current/",
            {"pk": 1, "username": "alice"},
            "username",
        ),
        (
            lambda api: api.model_details(7),
            "models/7/",
            dict(MODEL_JSON, pk=7),
            "name",
        ),
        (
            lambda api: api.document_details(42),
            "documents/42/",
            {"pk": 42, "name": "My Doc"},
            "name",
        ),
        (
            lambda api: api.task_details(9),
            "tasks/9/",
            dict(TASK_DATA, pk=9),
            "label",
        ),
    ],
    ids=["current-user", "model", "document", "task"],
)
def test_detail_endpoint_requests_item_and_parses_result(
    rsps, api_client_instance, call, path, payload, field
):
    rsps.get(f"{API_ROOT}/{path}", json=payload)

    result = call(api_client_instance)

    # pk and one more field, to check the result was parsed from this payload
    assert result.pk == payload["pk"]
    assert getattr(result, field) == payload[field]


# ---------------------------------------------------------------------------
# paged list endpoints
# ---------------------------------------------------------------------------

# Every paged list endpoint works the same way, so the tests below run for each.
# The lambdas make *which method to call* a test parameter; **kw passes on the
# optional `page` argument.
LIST_ENDPOINTS = pytest.mark.parametrize(
    ("call", "path", "result_type"),
    [
        (lambda api, **kw: api.model_list(**kw), "models/", "model"),
        (lambda api, **kw: api.document_list(**kw), "documents/", "document"),
        (
            lambda api, **kw: api.document_parts_list(42, **kw),
            "documents/42/parts/",
            "document_parts",
        ),
        (lambda api, **kw: api.task_list(**kw), "tasks/", "task"),
    ],
    ids=["models", "documents", "document-parts", "tasks"],
)


@LIST_ENDPOINTS
def test_list_endpoint_requests_first_page_by_default(
    rsps, api_client_instance, call, path, result_type
):
    rsps.get(f"{API_ROOT}/{path}", json=EMPTY_PAGE)

    result = call(api_client_instance)

    assert isinstance(result, ResultsList)
    assert result.result_type == result_type
    assert sent_query(rsps.calls[0]) == {}


@LIST_ENDPOINTS
def test_list_endpoint_passes_page(rsps, api_client_instance, call, path, result_type):
    rsps.get(f"{API_ROOT}/{path}", json=EMPTY_PAGE)

    call(api_client_instance, page=3)

    assert sent_query(rsps.calls[0]) == {"page": ["3"]}


def test_results_list_next_page_requests_next_url(rsps, api_client_instance):
    next_url = f"{API_ROOT}/models/?page=2"
    rsps.get(
        next_url,
        json={
            "count": 2,
            "next": None,
            "previous": f"{API_ROOT}/models/",
            "results": [dict(MODEL_JSON, pk=2)],
        },
    )
    page1 = ResultsList(
        api=api_client_instance,
        result_type="model",
        count=2,
        next=next_url,
        previous=None,
        results=[],
    )

    page2 = page1.next_page()

    assert page2.next is None
    # results of the next page are converted like those of the first
    assert page2.result_type == "model"
    assert [model.pk for model in page2.results] == [2]


def test_results_list_next_page_is_none_on_last_page(rsps, api_client_instance):
    last_page = ResultsList(
        api=api_client_instance,
        result_type="model",
        count=0,
        next=None,
        previous=None,
        results=[],
    )

    # no request is made: rsps has no responses registered, so one would fail
    assert last_page.next_page() is None


# ---------------------------------------------------------------------------
# models: create, update, delete
# ---------------------------------------------------------------------------


@pytest.fixture
def model_file(tmp_path):
    """A (fake) model file of 15 bytes."""
    model_file = tmp_path / "foo.mlmodel"
    model_file.write_bytes(b"fake model data")
    return model_file


@pytest.fixture
def model_accuracy(mocker):
    """Skip reading accuracy from a real kraken model file."""
    return mocker.patch("htr2hpc.api_client.get_model_accuracy", return_value=0.88)


def test_model_create_raises_on_invalid_job(api_client_instance, model_file):
    with pytest.raises(ValueError, match="not a valid model job name"):
        api_client_instance.model_create(model_file, job="Train")


def test_model_create_uploads_file_with_metadata(
    rsps, api_client_instance, model_file, model_accuracy
):
    # eScriptorium responds with 201 Created when a model is added
    rsps.post(f"{API_ROOT}/models/", status=201, json=dict(MODEL_JSON, name="foo"))

    result = api_client_instance.model_create(model_file, job="Recognize")

    assert result.name == "foo"
    assert sent_multipart(rsps.calls[0]) == {
        # model name defaults to the file name without its suffix
        "name": b"foo",
        "job": b"Recognize",
        "file_size": b"15",
        "training_accuracy": b"0.88",
        "file": b"fake model data",
    }
    model_accuracy.assert_called_once_with(model_file)


def test_model_create_uses_given_name(
    rsps, api_client_instance, model_file, model_accuracy
):
    rsps.post(f"{API_ROOT}/models/", status=201, json=MODEL_JSON)

    api_client_instance.model_create(model_file, job="Segment", model_name="custom")

    sent = sent_multipart(rsps.calls[0])
    assert sent["name"] == b"custom"
    assert sent["job"] == b"Segment"


def test_model_update_defaults_to_existing_job_and_name(
    rsps, api_client_instance, model_file, model_accuracy
):
    existing = dict(MODEL_JSON, pk=5, name="existing-name", job="Segment")
    rsps.get(f"{API_ROOT}/models/5/", json=existing)
    rsps.put(f"{API_ROOT}/models/5/", json=existing)

    result = api_client_instance.model_update(5, model_file)

    assert result.name == "existing-name"
    # first request looks up the model; the second replaces it
    sent = sent_multipart(rsps.calls[1])
    assert sent["name"] == b"existing-name"
    assert sent["job"] == b"Segment"
    assert sent["file_size"] == b"15"
    assert sent["training_accuracy"] == b"0.88"
    assert sent["file"] == b"fake model data"


def test_model_update_with_given_job_and_name_skips_lookup(
    rsps, api_client_instance, model_file, model_accuracy
):
    # only the update is registered: a lookup of the existing model would
    # be an unexpected request and fail
    rsps.put(f"{API_ROOT}/models/5/", json=MODEL_JSON)

    api_client_instance.model_update(
        5, model_file, job="Recognize", model_name="renamed"
    )

    sent = sent_multipart(rsps.calls[0])
    assert sent["name"] == b"renamed"
    assert sent["job"] == b"Recognize"


def test_model_delete(rsps, api_client_instance, caplog):
    # eScriptorium responds with 204 No Content when a model is deleted
    rsps.delete(f"{API_ROOT}/models/7/", status=204)

    api_client_instance.model_delete(7)

    # an unexpected status is only logged as an error, so check that none was
    assert "ERROR" not in [record.levelname for record in caplog.records]


# document_part_details


def part_json(lines):
    return {"pk": 5, "regions": [], "lines": lines}


def test_document_part_details_skips_lines_without_geometry(
    rsps, api_client_instance, caplog
):
    # eScriptorium can return lines with no baseline or mask;
    # kraken can't use these, so they must be dropped before training.
    # Line pks are integers in the real API.
    good = {"pk": 1, "mask": [[0, 0], [9, 9]], "baseline": [[0, 5], [9, 5]]}
    no_mask = {"pk": 2, "mask": None, "baseline": [[0, 5], [9, 5]]}
    no_baseline = {"pk": 3, "mask": [[0, 0], [9, 9]], "baseline": None}
    rsps.get(
        f"{API_ROOT}/documents/42/parts/5/",
        json=part_json([good, no_mask, no_baseline]),
    )

    with caplog.at_level(logging.WARNING, logger="htr2hpc.api_client"):
        part = api_client_instance.document_part_details(42, 5)

    assert [line.pk for line in part.lines] == [1]
    # the warning identifies which lines were skipped, and where they are
    assert "Skipping 2 lines" in caplog.text
    assert "2,3" in caplog.text
    assert "document 42, part 5" in caplog.text


def test_document_part_details_keeps_all_valid_lines(rsps, api_client_instance, caplog):
    good = {"pk": 1, "mask": [[0, 0], [9, 9]], "baseline": [[0, 5], [9, 5]]}
    rsps.get(f"{API_ROOT}/documents/42/parts/5/", json=part_json([good, good]))

    with caplog.at_level(logging.WARNING, logger="htr2hpc.api_client"):
        part = api_client_instance.document_part_details(42, 5)

    assert len(part.lines) == 2
    assert "Skipping" not in caplog.text


def test_document_part_details_returns_none_for_unparsable_part(
    rsps, api_client_instance
):
    # a field name that can't become a namedtuple attribute; the part is
    # skipped (None) so that one bad part doesn't abort a whole training run
    rsps.get(
        f"{API_ROOT}/documents/42/parts/5/",
        json={"pk": 5, "lines": [], "not a valid name": 1},
    )

    assert api_client_instance.document_part_details(42, 5) is None


# document_part_transcription_list


@pytest.mark.parametrize(
    ("kwargs", "expected_query"),
    [({"transcription_id": 7}, {"transcription": ["7"]}), ({}, {})],
    ids=["filtered", "unfiltered"],
)
def test_document_part_transcription_list_filters_by_transcription(
    rsps, api_client_instance, kwargs, expected_query
):
    rsps.get(f"{API_ROOT}/documents/42/parts/5/transcriptions/", json=EMPTY_PAGE)

    api_client_instance.document_part_transcription_list(42, 5, **kwargs)

    # without the filter we would get lines from every transcription
    assert sent_query(rsps.calls[0]) == expected_query


# document_export


def test_document_export_requests_alto_with_all_region_types(rsps, api_client_instance):
    # region types come from the document's own block types, plus the two
    # pseudo-types eScriptorium uses for untyped and unattached regions
    rsps.get(
        f"{API_ROOT}/documents/42/",
        json={"pk": 42, "valid_block_types": [{"pk": 1}, {"pk": 2}]},
    )
    rsps.post(f"{API_ROOT}/documents/42/export/", json={"status": "ok"})

    result = api_client_instance.document_export(42, transcription_id=7)

    assert result.status == "ok"
    form = sent_form(rsps.calls[1])
    assert form["file_format"] == ["alto"]
    assert form["transcription"] == ["7"]
    assert form["include_images"] == ["False"]
    assert form["region_types"] == ["1", "2", "Undefined", "Orphan"]


# list_types


def test_list_types_rejects_unsupported_type(api_client_instance):
    with pytest.raises(ValueError, match="not a supported type"):
        api_client_instance.list_types("document")


def test_list_types_requests_type_endpoint(rsps, api_client_instance):
    rsps.get(f"{API_ROOT}/types/block/", json=EMPTY_PAGE)

    result = api_client_instance.list_types("block")

    assert isinstance(result, ResultsList)
    assert result.result_type == "type"


# download_file


@pytest.mark.parametrize(
    ("explicit_name", "disposition", "expected_name"),
    [
        # a caller-supplied name always wins
        ("chosen.zip", 'attachment; filename="srv.zip"', "chosen.zip"),
        # otherwise use the server's suggested name
        (None, 'attachment; filename="srv.zip"', "srv.zip"),
        # with no hint at all, fall back to the end of the url
        (None, None, "file.zip"),
    ],
    ids=["explicit-name", "content-disposition", "url-fallback"],
)
def test_download_file_chooses_filename(
    rsps, api_client_instance, tmp_path, explicit_name, disposition, expected_name
):
    url = "https://escriptorium.example.com/media/file.zip"
    body = b"zip content"
    # download_file reads content-length to log the size of the download (and
    # fails without it), so set it like a real server does
    headers = {"content-length": str(len(body))}
    if disposition:
        headers["content-disposition"] = disposition
    rsps.get(url, body=body, headers=headers)

    saved = api_client_instance.download_file(url, tmp_path, filename=explicit_name)

    assert saved == tmp_path / expected_name
    assert saved.read_bytes() == body


def test_download_file_returns_none_on_error_status(
    rsps, api_client_instance, tmp_path
):
    url = "https://escriptorium.example.com/media/missing.zip"
    rsps.get(url, status=404)

    assert api_client_instance.download_file(url, tmp_path) is None
    # nothing should be left behind for later steps to trip over
    assert list(tmp_path.iterdir()) == []


# download_document_export


def task_json(**overrides):
    """A task API response, based on the TASK_DATA fixture above."""
    return dict(TASK_DATA, **overrides)


def test_download_document_export_waits_for_task_then_downloads(
    rsps, mocker, api_client_instance, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # the export is saved in the working directory
    rsps.get(
        f"{API_ROOT}/documents/42/",
        json={"pk": 42, "valid_block_types": []},
    )
    rsps.post(f"{API_ROOT}/documents/42/export/", json={"status": "ok"})
    # the newest task is listed first, and is not finished yet
    queued = task_json(pk=99, done_at=None)
    rsps.get(
        f"{API_ROOT}/tasks/",
        json=dict(EMPTY_PAGE, count=1, results=[queued]),
    )
    # the first status check is still running; the second is done
    rsps.get(f"{API_ROOT}/tasks/99/", json=queued)
    rsps.get(f"{API_ROOT}/tasks/99/", json=task_json(pk=99))
    # file name is based on the time the task finished (see TASK_DATA)
    export_url = (
        "https://escriptorium.example.com/media/users/3/"
        "export_doc42_my_doc_alto_202401151001.zip"
    )
    rsps.get(
        export_url,
        body=b"zipped alto",
        headers={
            "content-disposition": 'attachment; filename="export.zip"',
            "content-length": "11",
        },
    )

    mock_sleep = mocker.patch("htr2hpc.api_client.sleep")
    saved = api_client_instance.download_document_export(
        user_id=3, document_id=42, document_name="My Doc", transcription_ids=7
    )

    # waits one second each time it finds the task still unfinished:
    # once for the task from the list, once for the first status check
    assert mock_sleep.call_count == 2
    # returned path is relative to the working directory (set to tmp_path above)
    assert saved.name == "export.zip"
    assert (tmp_path / saved).read_bytes() == b"zipped alto"


# task_update


def test_task_update_sends_label_user_and_messages(rsps, api_client_instance):
    rsps.put(f"{API_ROOT}/tasks/9/", json=task_json(pk=9))

    api_client_instance.task_update(9, label="Train", user_id=3, messages="step 2")

    assert sent_form(rsps.calls[0]) == {
        "label": ["Train"],
        "user": ["3"],
        "messages": ["step 2"],
    }


def test_task_update_omits_messages_when_not_given(rsps, api_client_instance):
    rsps.put(f"{API_ROOT}/tasks/9/", json=task_json(pk=9))

    api_client_instance.task_update(9, label="Train", user_id=3)

    # leaving messages out keeps any existing messages on the task
    assert "messages" not in sent_form(rsps.calls[0])
