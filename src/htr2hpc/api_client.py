import datetime
import logging
from collections import namedtuple
from dataclasses import dataclass
from time import sleep
import pathlib
from typing import Optional, Any
import json

import humanize
import requests
from django.utils.text import slugify  # is django a reasonable dependency?
import coremltools

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

    # NOTE: workflow state is a numeric id defined in escriptorium
    # current values:
    # WORKFLOW_STATE_QUEUED = 0
    # WORKFLOW_STATE_STARTED = 1
    # WORKFLOW_STATE_ERROR = 2
    # WORKFLOW_STATE_DONE = 3
    # WORKFLOW_STATE_CANCELED = 4
    # add a property?

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


@dataclass
class Workflow:
    convert: Optional[str] = None
    segment: Optional[str] = None
    # workflow status is only present when a workflow has not run,
    # so define a dataclass and make them optional
    # to handle missing values


# keep a registry of result classes for API result objects,
# so they can be reused once they are defined
RESULTCLASS_REGISTRY = {"task": Task, "workflow": Workflow}


def to_namedtuple(name: str, data: Any):
    """convenience method to convert API response data into namedtuple objects
    for easier attribute access"""

    # when data is a dictionary, convert to a namedtuple with the specified name
    if isinstance(data, dict):
        # if a namedtuple already exists for this name, use it
        nt_class = RESULTCLASS_REGISTRY.get(name)
        # otherwise, create it and add it to the registry
        if nt_class is None:
            logger.debug(f"Creating namedtuple with name {name}")
            nt_class = namedtuple(name, data)
            RESULTCLASS_REGISTRY[name] = nt_class
        else:
            logger.debug(f"Using existing result class for {name}: {nt_class}")
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
    def __init__(self, base_url: str, api_token: str):
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

    def _make_request(
        self,
        url: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        files: Optional[dict] = None,
        method: str = "GET",
        expected_status: int = requests.codes.ok,
    ):
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
        elif method in ["POST", "PUT"]:
            session_request = getattr(self.session, method.lower())
            # add post data and files to the request if any are specified
            if data:
                rqst_opts["data"] = data
            if files:
                rqst_opts["files"] = files
        else:
            raise ValueError(f"unsupported http method: {method}")

        resp = session_request(rqst_url, **rqst_opts)
        logger.debug(
            f"get {rqst_url} {resp.status_code}: {resp.elapsed.total_seconds()} sec"
        )
        if resp.status_code == expected_status:
            return resp

        # disable for now, this does not work in all cases
        # elif resp.status_code == requests.codes.bad_request:
        #     try:
        #         info = to_namedtuple("bad_request", resp.json())
        #         details = ""
        #         if info.status == "error":
        #             details = info.error
        #         logger.error(f"Bad request {details}")
        #     except AttributeError:
        #         print(resp.content)
        elif resp.status_code == requests.codes.not_found:
            logger.error("Error: not found")
        elif resp.status_code == requests.codes.unauthorized:
            logger.error("Error: not authorized")
        else:
            # TODO: error handling / logging for other cases?
            # (useful for dev if nothing else...)
            logger.error(resp.status_code)
            logger.error(resp.content)
            resp.raise_for_status()

    def get_current_user(self):
        """Get information about the current user account"""
        api_url = "users/current/"
        return to_namedtuple("user", self._make_request(api_url).json())

    def model_list(self):
        """paginated list of models"""
        api_url = "models/"
        resp = self._make_request(api_url)
        return ResultsList(result_type="list_model", **resp.json())

    def model_details(self, model_id):
        """details for a single models"""
        api_url = f"models/{model_id}/"
        resp = self._make_request(api_url)
        return to_namedtuple("model", resp.json())

    def model_update(self, model_id: int, model_file: pathlib.Path):
        """Update an existing model record with a new model file."""
        api_url = f"models/{model_id}/"
        with open(model_file, "rb") as mfile:
            files = {"file": mfile}
            data = {
                # TODO don't overwrite previous name!
                "name": "updated model",
                "job": "Segment",  # required; get from existing record?
                # file_size (int)  - set from pathlib object using .stat().st_size
                # versions ?
                # accuracy_percent
            }
            resp = self._make_request(api_url, method="PUT", files=files, data=data)
        # on successful update, returns the model object
        return to_namedtuple("model", resp.json())

    def model_create(
        self,
        model_file: pathlib.Path,
        job: str,
        model_name: Optional[str] = None,
    ):
        """Add a new model to eScriptorium. Takes a model file, name, and job
        (Segment or Recognize)."""
        if job not in {"Segment", "Recognize"}:
            raise ValueError(f"{job} is not a valid model job name")

        api_url = "models/"
        # if model name is unset, use filename stem
        if model_name is None:
            model_name = model_file.stem

        with open(model_file, "rb") as mfile:
            files = {"file": mfile}
            data = {
                # NOTE: could optionally infer model name from filename
                "name": model_name,
                "job": job,
                "file_size": model_file.stat().st_size,
                # get accuracy from model
                # "accuracy_percent": self.get_model_accuracy(model_file),
                # NOTE: eScriptorium api has accuracy marked as a read-only
                # field, so even if we supply this it gets set to zero.
                # Calculating it is slow, so skip since it isn't currently usable
            }
            resp = self._make_request(
                api_url,
                method="POST",
                files=files,
                data=data,
                # should return 201 Created on success
                expected_status=requests.codes.created,
            )
        # on successful update, returns the model object
        return to_namedtuple("model", resp.json())

    def get_model_accuracy(self, model_file: pathlib.Path):
        m = coremltools.models.MLModel(str(model_file))
        meta = json.loads(m.get_spec().description.metadata.userDefined["kraken_meta"])
        return meta["accuracy"][-1][-1] * 100

    def document_list(self):
        """paginated list of documents"""
        api_url = "documents/"
        resp = self._make_request(api_url)
        return ResultsList(result_type="document", **resp.json())

    def document_details(self, document_id: int):
        """details for a single document"""
        api_url = f"documents/{document_id}/"
        resp = self._make_request(api_url)
        return to_namedtuple("document", resp.json())

    def document_parts_list(self, document_id: int):
        """list of all the parts associated with a document"""
        api_url = f"documents/{document_id}/parts/"
        resp = self._make_request(api_url)
        # document part listed here is different than full parts result
        return ResultsList(result_type="documentpart", **resp.json())

    def document_part_details(self, document_id: int, part_id: int):
        """details for one part of a document"""
        api_url = f"documents/{document_id}/parts/{part_id}/"
        resp = self._make_request(api_url)
        return to_namedtuple("part", resp.json())

    def document_part_transcription_list(
        self, document_id: int, part_id: int, transcription_id: Optional[int] = None
    ):
        """list of transcription lines for one part of a document"""
        api_url = f"documents/{document_id}/parts/{part_id}/transcriptions/"
        params = {}
        # if transcription id is specified, pass as a querystring param to filter
        # to only lines from that specific transcription
        if transcription_id is not None:
            params["transcription"] = transcription_id
        resp = self._make_request(api_url, params=params)
        return ResultsList(result_type="transcription_line", **resp.json())

    def document_export(
        self, document_id: int, transcription_id: int, include_images: bool = False
    ):
        """request a document export be compiled for download"""
        api_url = f"documents/{document_id}/export/"
        # export form requires a region_types list, which
        # is a based on block type ids for this document
        # and may also include "Undefined" and "Orphan"
        document_info = self.document(document_id)
        block_types = [block.pk for block in document_info.valid_block_types]
        block_types.extend(["Undefined", "Orphan"])

        data = {
            "transcription": transcription_id,  # could be multiple(?)
            "file_format": "alto",
            "include_images": include_images,
            "region_types": block_types,
        }

        resp = self._make_request(api_url, method="POST", data=data)

        # NOTE: specifying a valid document id and invalid transription id
        # returns a 400 bad request
        return to_namedtuple("status", resp.json())

    # API provides types for these items
    types = ["block", "line", "annotations", "part"]

    def list_types(self, item):
        """list of available types"""
        if item not in self.types:
            raise ValueError(f"{item} is not a supported type for list types")
        api_url = f"types/{item}/"
        resp = self._make_request(api_url)
        return ResultsList(result_type="type", **resp.json())

    def export_file_url(
        self,
        user_id: int,
        document_id: int,
        document_name: str,
        file_format: str,
        creation_time: datetime.datetime,
    ) -> str:
        """Generate URL for an export file generated by :meth:`document_export`"""
        # is there any way get current user id from api based purely on token?
        # if not, consider using users api endpoint to get numeric id from username

        # ... get document name from api by document id?

        # NOTE: filename logic copied & adapted from escriptorium
        # import.export.BaseExporter logic
        # NOTE2: escriptorium code uses datetime.now() so there's no guarantee
        # this will match the completion time of the task...
        base_filename = "export_doc%d_%s_%s_%s" % (
            document_id,
            slugify(document_name).replace("-", "_")[:32],
            file_format,
            creation_time.strftime("%Y%m%d%H%M"),
            # creation_time.strftime("%Y%m%d%H%M%S"),
        )

        return f"{self.base_url}/media/users/{user_id}/{base_filename}.zip"

    def download_document_export(
        self, user_id, document_id, document_name, transcription_ids
    ) -> pathlib.Path:
        """Request a document export, monitor the task until it completes,
        then download the compiled document export file.
        """

        # use document_export api method to start an async export task
        # NOTE: several more options should be configurable
        result = self.document_export(
            document_id=document_id,
            transcription_id=transcription_ids,
            include_images=True,
        )
        # currently returns status=ok, indicating a task has been queued
        if result.status == "ok":
            task_list = self.task_list()
            # assume most recent task, which is listed first
            # TODO: confirm correct task id based on method (export) and document name
            export_task = task_list.results[0]

        # until the task completes, sleep and then check status
        # TODO: use task.workflow_state ?
        # TODO: handle canceled task
        while export_task.done_at is None:
            sleep(1)
            # refresh task details from api to check status
            export_task = self.task_details(export_task.pk)

        logger.info(f"Export XML completed after {export_task.duration()}")
        export_file_url = self.export_file_url(
            user_id,
            document_id,
            document_name,
            "alto",
            export_task.done_at
            # user_id, document_id, document_name, "alto", export_task.created_at
        )
        logger.info(f"Downloading export from {export_file_url}")

        resp = requests.get(export_file_url, stream=True)
        if resp.status_code == requests.codes.ok:
            # content disposition includes filename
            content_disposition = resp.headers.get("content-disposition")
            content_length = resp.headers.get("content-length")
            filename = content_disposition.split("filename=")[-1].strip('"')
            outfile = pathlib.Path(filename)
            # report on filename and size based on content-length header
            logger.info(
                f"Saving as {filename} ({humanize.naturalsize(content_length)})"
            )
            with outfile.open("wb") as filehandle:
                for chunk in resp.iter_content(chunk_size=1024):
                    filehandle.write(chunk)
            return outfile

    def download_file(
        self, url: str, save_location: pathlib.Path, filename=None
    ) -> Optional[pathlib.Path]:
        """Convenience method to download a file to a specified location.
        Returns"""

        # TODO: would be nice to make url relative to base_url if it's
        # a relative url and not a full path
        resp = requests.get(url, stream=True)
        if resp.status_code == requests.codes.ok:
            content_length = resp.headers.get("content-length")

            if not filename:
                # content disposition may include filename
                content_disposition = resp.headers.get("content-disposition")
                if content_disposition:
                    filename = content_disposition.split("filename=")[-1].strip('"')
                else:
                    # if filename is not set in header, use the end of the url
                    filename = url.rsplit("/")[-1]

            outfile = save_location / filename
            # report on filename and size based on content-length header
            logger.info(
                f"Saving as {filename} ({humanize.naturalsize(content_length)})"
            )
            with outfile.open("wb") as filehandle:
                for chunk in resp.iter_content(chunk_size=1024):
                    filehandle.write(chunk)
            return outfile

        return None

    def task_list(self):
        """paginated list of tasks"""
        api_url = "tasks/"
        resp = self._make_request(api_url)
        return ResultsList(result_type="task", **resp.json())

    def task_details(self, task_id: int):
        """details for a single task"""
        api_url = f"tasks/{task_id}/"
        resp = self._make_request(api_url)
        return to_namedtuple("task", resp.json())
