import datetime
import logging
from collections import namedtuple
from dataclasses import dataclass
from time import sleep

import requests
from django.utils.text import slugify  # is django a reasonable dependency?

from htr2hpc import __version__ as _version

logger = logging.getLogger(__name__)


@dataclass
class ResultsList:
    """API list response."""

    # all API list methods have the same structure,
    # so use a dataclass but specify the result type

    result_type: str
    count: int
    next: str
    previous: str
    results: list

    def __post_init__(self):
        # convert result entries to namedtuple class based on
        # specified result type name
        self.results = [to_namedtuple(self.result_type, d) for d in self.results]


@dataclass
class Task:
    """API response for a task result."""

    pk: int
    document: int
    document_part: int
    workflow_state: int
    label: str
    messages: str
    queued_at: datetime.datetime
    started_at: datetime.datetime
    done_at: datetime.datetime
    method: str
    user: int

    def __post_init__(self):
        # convert dates from string to datetime
        for date_field in ["queued_at", "started_at", "done_at"]:
            value = getattr(self, date_field)
            if value:
                setattr(self, date_field, datetime.datetime.fromisoformat(value))

    def duration(self) -> datetime.timedelta | None:
        "how long the task took to complete"
        if self.done_at is not None:
            return self.done_at - self.started_at


# keep a registry of result classes for API result objects,
# so they can be reused once they are defined
RESULTCLASS_REGISTRY = {
    "task": Task,
}


def to_namedtuple(name: str, data: any):
    """convenience method to convert API response data into namedtuple objects
    for easier attribute access"""

    # when data is a dictionary, convert to a namedtuple with the specified name
    if isinstance(data, dict):
        # if a namedtuple already exists for this name, use it
        nt_class = RESULTCLASS_REGISTRY.get(name)
        # otherwise, create it and add it to the registry
        if nt_class is None:
            nt_class = namedtuple(name, data)
            RESULTCLASS_REGISTRY[name] = nt_class
        # once we have the class, initialize an instance with the given data
        return nt_class(
            # convert any nested objects to namedtuple classes
            # use key name as namedtuple class name; convert plurals to singular
            # NOTE: could add aliases other cleanup, e.g. shared_with_user(s) == user
            **{key: to_namedtuple(key.rstrip("s"), val) for key, val in data.items()}
        )

    # when data is a list, convert each item in the list
    # to a namedtuple with the specified name
    elif isinstance(data, list):
        return [to_namedtuple(name, d) for d in data]

    # for all other data types, return as-is
    else:
        return data


class eScriptoriumAPIClient:
    def __init__(self, base_url, api_token):
        # ensure no trailing slash before combining with other urls
        self.base_url = base_url.rstrip("/")
        self.api_root = f"{self.base_url}/api"
        self.api_token = api_token
        # create a request session, for request pooling
        self.session = requests.Session()
        headers = {
            # set a user-agent header, but  preserve requests version information
            "User-Agent": f"htr2hpc/{_version} ({self.session.headers['User-Agent']})",
            "Authorization": f"Token {self.api_token}",
        }
        self.session.headers.update(headers)

    def _make_request(self, url, params=None, data=None, method="GET"):
        """
        Make a GET request with the configured session. Takes a url
        relative to :attr:`api_root` and optional dictionary of parameters for the request.
        """
        rqst_url = f"{self.api_root}/{url}"
        rqst_opts = {}
        if params:
            rqst_opts["params"] = params.copy()

        if method == "GET":
            session_request = self.session.get
        elif method == "POST":
            session_request = self.session.post
            # add post data to the request if there is any
            if data:
                rqst_opts["data"] = data
        else:
            raise ValueErorr(f"unsupported http method: {method}")

        resp = session_request(rqst_url, **rqst_opts)
        logger.debug(
            "get %s %s: %f sec",
            rqst_url,
            resp.status_code,
            resp.elapsed.total_seconds(),
        )
        if resp.status_code == requests.codes.ok:
            return resp
        if resp.status_code == requests.codes.not_found:
            print("Error: not found")

        if resp.status_code == requests.codes.unauthorized:
            print("Error: not authorized")

    def models(self):
        """paginated list of models"""
        api_url = "models/"
        resp = self._make_request(api_url)
        return ResultsList(result_type="model", **resp.json())

    def documents(self):
        """paginated list of documents"""
        api_url = "documents/"
        resp = self._make_request(api_url)
        return ResultsList(result_type="document", **resp.json())

    def document(self, document_id):
        """details for a single document"""
        api_url = f"documents/{document_id}/"
        resp = self._make_request(api_url)
        return to_namedtuple("document", resp.json())

    def document_export(self, document_id, transcription_id):
        """request a document export be compiled for download"""
        api_url = f"documents/{document_id}/export/"
        # export form requires a region_types list, which
        # is a based on block type ids for this document
        # and may also include "Undefined" and "Orphan"
        document_info = self.document(document_id)
        block_types = [block.pk for block in document_info.valid_block_types]
        block_types.extend(["Undefined", "Orphan"])

        data = {
            "transcription": transcription_id,
            "file_format": "alto",
            "include_images": False,
            "region_types": block_types,
        }

        resp = self._make_request(api_url, method="POST", data=data)
        return to_namedtuple("status", resp.json())

    def export_file_url(
        self, user_id, document_id, document_name, file_format, creation_time
    ):
        # is it possible to get user id from api based purely on token?
        # if not, consider using users api endpoint to get numeric id from username

        # ... could get document name from api by document id

        # NOTE: filename logic copied & adapted from escriptorium
        # import.export.BaseExporter logic
        base_filename = "export_doc%d_%s_%s_%s" % (
            document_id,
            slugify(document_name).replace("-", "_")[:32],
            file_format,
            creation_time.strftime("%Y%m%d%H%M%S"),
        )

        return f"{self.base_url}/media/users/{user_id}/{base_filename}.zip"

    def tasks(self):
        """paginated list of tasks"""
        api_url = "tasks/"
        resp = self._make_request(api_url)
        return ResultsList(result_type="task", **resp.json())

    def task(self, task_id):
        """details for a single task"""
        api_url = f"tasks/{task_id}/"
        resp = self._make_request(api_url)
        return to_namedtuple("task", resp.json())
