#!/usr/bin/env python
import argparse
import os
import sys
import logging
import pathlib
import time
from dataclasses import dataclass
from shutil import rmtree
from typing import Optional

from intspan import intspan
from kraken.kraken import SEGMENTATION_DEFAULT_MODEL, DEFAULT_MODEL
from tqdm import tqdm

from htr2hpc.api_client import eScriptoriumAPIClient, NotFound, NotAllowed
from htr2hpc.train.data import get_training_data, get_model, upload_models
from htr2hpc.train.slurm import segtrain, slurm_job_status, slurm_job_queue_status


api_token_env_var = "ESCRIPTORIUM_API_TOKEN"

# map our job type option choices to escriptorium terms
es_model_jobs = {"segmentation": "Segment", "transcription": "Recognize"}

# kraken python package provides paths to the default best models for both modes
default_model = {
    "segmentation": SEGMENTATION_DEFAULT_MODEL,
    "transcription": DEFAULT_MODEL,
}


@dataclass
class TrainingManager:
    base_url: str
    api_token: str
    work_dir: pathlib.Path
    document_id: int
    training_mode: str
    model_name: str
    num_workers: int
    parts: Optional[intspan] = None
    model_id: Optional[int] = None
    existing_data: bool = False
    show_progress: bool = True

    def __post_init__(self):
        # initialize api client
        self.api = eScriptoriumAPIClient(self.base_url, self.api_token)

        # store the path to original working directory before changing directory
        self.orig_working_dir = pathlib.Path.cwd()

    def training_prep(self):
        # create necessary directories and download training data and model file

        self.training_data_dir = self.work_dir / "parts"
        if not self.existing_data:
            self.training_data_dir.mkdir()
        get_training_data(
            self.api, self.training_data_dir, self.document_id, self.parts
        )

        # if model id is specified, download the model from escriptorium API,
        # confirming that it is the appropriate type (segmentation/transcription)
        # NOTE: currently ignores existing data flag, since we need model file name
        if self.model_id:
            self.model_file = get_model(
                self.api,
                self.model_id,
                self.training_mode,
                self.work_dir,
            )
        # if model id is not specified, use the default from kraken
        else:
            # get the appropriate model file for the requested training mode
            # kraken default defs are path objects
            self.model_file = default_model[args.mode]

        # create a directory and path for the output model file
        self.output_model_dir = self.work_dir / "output_model"
        # remove the output model directory to avoid confusion with any old
        # model files from a previous run
        if self.existing_data:
            rmtree(self.output_model_dir)
        self.output_model_dir.mkdir()
        self.output_modelfile = self.output_model_dir / self.model_name

    def monitor_slurm_job(self, job_id):
        # get initial job status (typically PENDING)
        job_status = slurm_job_queue_status(job_id)
        # typical states are PENDING, RUNNING, SUSPENDED, COMPLETING, and COMPLETED.
        # https://slurm.schedmd.com/job_state_codes.html
        # end states could be FAILED, CANCELLED, OUT_OF_MEMORY, TIMEOUT
        # * but note that squeue only reports on pending & running jobs

        # loop while the job is pending or running and then stop
        # use tqdm to display job status and wait time
        with tqdm(
            desc=f"Slurm job {job_id}",
            bar_format="{desc} | total time: {elapsed}{postfix} ",
            disable=not args.show_progress,
        ) as statusbar:
            running = False
            while job_status:
                status = f"status: {job_status}"
                # display an unofficial runtime to aid in troubleshooting
                if running:
                    runtime_elapsed = statusbar.format_interval(time.time() - runstart)
                    status = f"{status}  ~ run time: {runtime_elapsed}"
                statusbar.set_postfix_str(status)
                time.sleep(1)
                job_status = slurm_job_queue_status(job_id)
                # capture start time first time we get a status of running
                if not running and job_status == "RUNNING":
                    running = True
                    runstart = time.time()

        # check the completed status
        job_status = slurm_job_status(job_id)
        print(
            f"Job {job_id} is no longer queued; ending status: {','.join(job_status)}"
        )
        job_output = args.work_dir / f"segtrain_{job_id}.out"
        print(f"Job output is in {job_output}")

    def segmentation_training(self):
        # get absolute versions of these paths _before_ changing working directory
        abs_training_data_dir = self.training_data_dir.absolute()
        abs_model_file = self.model_file.absolute()
        abs_output_modelfile = self.output_modelfile.absolute()

        # change directory to working directory, since by default,
        # slurm executes the job from the directory where it was submitted
        os.chdir(self.work_dir)

        job_id = segtrain(
            abs_training_data_dir,
            abs_model_file,
            abs_output_modelfile,
            self.num_workers,
        )
        self.monitor_slurm_job(job_id)
        # change back to original working directory
        os.chdir(self.orig_working_dir)
        self.upload_models()

    def upload_models(self):
        # - for segmentation, upload all models to eScriptorium as new models
        upload_count = upload_models(
            self.api,
            self.output_modelfile.parent,
            self.training_mode,
            show_progress=self.show_progress,
        )
        # - should this behavior depend on job exit status?
        # reasonable to assume any model files created should be uploaded?
        print(f"Uploaded {upload_count} {self.training_mode} models to eScriptorium")


def main():
    try:
        api_token = os.environ[api_token_env_var]
    except KeyError:
        print(
            f"Error: eScriptorium API token must be set as environment variable {api_token_env_var}",
            file=sys.stderr,
        )
        sys.exit(1)

    # TODO: add options for:
    # - create/update model flag (make explicit)
    #   name for the new model when creating a new one (required)

    # use subparsers for the two modes
    parser = argparse.ArgumentParser(
        description="Export content from eScriptorium and train or fine-tune models"
    )
    subparsers = parser.add_subparsers(
        title="mode", description="supported training modes", required=True, dest="mode"
    )
    subparsers.add_parser("segmentation")  # currently no segmentation-specific options
    transcription_parser = subparsers.add_parser("transcription")

    # common arguments used in both modes
    parser.add_argument(
        "base_url",
        metavar="BASE_URL",
        help="Base URL for eScriptorium instance (without /api/)",
        type=str,
    )
    parser.add_argument(
        "work_dir",
        metavar="WORKING_DIR",
        help="Working directory where data should be downloaded (must NOT already exist)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-d",
        "--document",
        help="Document id to export",
        type=int,
        dest="document_id",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Optional model id to use for fine-tuning",
        type=int,
        dest="model_id",
    )
    parser.add_argument(
        "--model-name",
        help="Name to be used for the newly trained model",
        type=str,
        dest="model_name",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--parts",
        help="Optional list of part ids for training. Format as #,#,#  or #-##."
        + "(if not specified, uses entire document)",
        type=intspan,
    )
    parser.add_argument(
        "--existing-data",
        help="Use existing data from a previous run",
        action="store_true",
        default=False,
    )
    # control whether or not to clean up temporary files (on by default)
    parser.add_argument(
        "--clean",
        help="Clean up temporary working files after training ends",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # control progress bar display (on by default)
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="show_progress",
    )

    # training for transcription requires a transcription id
    transcription_parser.add_argument(
        "-t",
        "--transcription",
        help="Transcription id to export",
        type=int,
        dest="transcription_id",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--workers",
        help="Number of workers for training task (default: %(default)d)",
        type=int,
        default=8,
        dest="num_workers",
    )
    args = parser.parse_args()

    # bail out on transcription for now (will be added later)
    if args.mode == "transcription":
        print("Transcription training is not yet supported")
        sys.exit(1)

    # make sure working directory does not already exist
    if args.work_dir.exists() and not args.existing_data:
        print(
            f"Working directory `{args.work_dir}` already exists (use --existing-data to allow)",
            file=sys.stderr,
        )
        # NOTE: existing-data option allows reusing previously downloaded data, but this
        # is primarily a dev/test workaround, does not handle all cases
        sys.exit(1)
    if args.existing_data and not args.work_dir.exists():
        print(
            f"Working directory `{args.work_dir}` does not exist but --existing-data was requested",
            file=sys.stderr,
        )
        sys.exit(1)

    # create new working directory if it doesn't already exist
    if not args.existing_data:
        args.work_dir.mkdir()

    logging.basicConfig(encoding="utf-8", level=logging.WARN)
    logger_upscope = logging.getLogger("htr2hpc")
    logger_upscope.setLevel(logging.INFO)

    # TODO : check api access works before going too far?
    # (currently does not handle connection error gracefully)

    # nearly all the argparse options need to be passed to the training manager class
    # convert to a _copy_ dictionary and delete the unused parmeters
    arg_options = dict(vars(args))
    del arg_options["mode"]
    del arg_options["clean"]

    # initialize training manager
    training_mgr = TrainingManager(
        api_token=api_token, training_mode=es_model_jobs[args.mode], **arg_options
    )
    try:
        # prep data for training
        training_mgr.training_prep()

        if args.mode == "segmentation":
            training_mgr.segmentation_training()
    except (NotFound, NotAllowed) as err:
        print(f"Something went wrong: {err}")

    # TODO: handle transcription training

    # unless requested not to, clean up the working directory, which includes:
    # - downloaded training data & model to fine tune
    # - generated models
    # - training output
    if args.clean:
        print(
            f"Removing working directory {args.work_dir} with all training data and models."
        )
        rmtree(args.work_dir)

    # when this is all working, cleanup working dir (by default, with option to skip)


if __name__ == "__main__":
    main()
