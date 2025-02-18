#!/usr/bin/env python
import argparse
import os
import sys
from enum import Enum

import logging
import pathlib
import time
from dataclasses import dataclass
from shutil import rmtree
from typing import Optional

from intspan import intspan
from kraken.kraken import SEGMENTATION_DEFAULT_MODEL
import requests
from tqdm import tqdm
from urllib3.exceptions import HTTPError

# from urllib3.exceptions import ConnectionError

from htr2hpc.api_client import eScriptoriumAPIClient, NotFound, NotAllowed
from htr2hpc.train.data import (
    get_training_data,
    get_model_file,
    upload_models,
    upload_best_model,
    get_prelim_model,
)
from htr2hpc.train.slurm import (
    segtrain,
    slurm_job_status,
    slurm_job_queue_status,
    slurm_job_stats,
    recognition_train,
)
from htr2hpc.train.calculate import (
    slurm_get_max_acc,
    calc_full_duration,
    calc_cpu_mem,
    estimate_duration,
    estimate_cpu_mem,
)


api_token_env_var = "ESCRIPTORIUM_API_TOKEN"

# map our job type option choices to escriptorium terms
es_model_jobs = {"segmentation": "Segment", "transcription": "Recognize"}


class JobCancelled(Exception):
    "Custom exception for when slurm job was cancelled"


class UpdateMode(Enum):
    NEVER = 0
    ALWAYS = 1
    IF_IMPROVED = 2

    def __bool__(self):
        # override so boolean check of never will evaluate to false
        return self != UpdateMode.NEVER


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
    task_report_id: Optional[int] = None
    update: UpdateMode = UpdateMode.NEVER
    transcription_id: Optional[int] = None
    existing_data: bool = False
    show_progress: bool = True
    model_file: pathlib.Path = None
    training_data_counts: Optional[dict] = None
    slurm_output: str = ""
    job_stats: str = ""

    def __post_init__(self):
        if self.update and not self.model_id:
            raise ValueError("Cannot set update to true if model_id is not set")

        # initialize api client
        self.api = eScriptoriumAPIClient(self.base_url, self.api_token)
        # Report on current user, to confirm the expected account is in use.
        # This also serves as a configuration check before going further.
        try:
            current_user = self.api.get_current_user()
            print(
                f"Connecting to eScriptorium as {current_user.username} ({current_user.email})"
            )
        except (requests.exceptions.ConnectionError, NotFound, NotAllowed) as err:
            # invalid hostname raises a connection error
            # wrong hostname (no API endpoint) raises not found api error
            raise ConnectionError(f"Error connecting to eScriptorium: {err}")

        # store the path to original working directory before changing directory
        self.orig_working_dir = pathlib.Path.cwd()

    def training_prep(self):
        # create necessary directories and download training data and model file

        self.training_data_dir = self.work_dir / "parts"
        if not self.existing_data:
            self.training_data_dir.mkdir()

        # get training data and store the counts of number of parts, regions, lines
        self.training_data_counts = get_training_data(
            self.api,
            self.training_data_dir,
            self.document_id,
            self.parts,
            self.transcription_id,
        )

        # if model id is specified, download the model from escriptorium API,
        # confirming that it is the appropriate type (segmentation/transcription)
        # NOTE: currently ignores existing data flag, since we need model file name
        # TODO: handle model with no file (i.e., newly created model in eScriptorium)
        if self.model_id:
            # when model id + update are specified,
            # use model name from api info
            if self.model_name is None:
                model_info = self.api.model_details(self.model_id)
                # NOTE: model name does not necessarily match filename
                # exactly, e.g. bnSEG_complex vs bnseg_complex
                self.model_name = model_info.name

            self.model_file = get_model_file(
                self.api,
                self.model_id,
                self.training_mode,
                self.work_dir,
            )

        # if model id is not specified or model id has no file
        # and we are doing segmentation training, use the default from kraken
        if self.training_mode == "Segment" and not self.model_file:
            self.model_file = SEGMENTATION_DEFAULT_MODEL

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
            disable=not self.show_progress,
        ) as statusbar:
            running = False
            runstart = time.time()
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
        if self.training_mode == "Segment":
            job_output = self.work_dir / f"segtrain_{job_id}.out"
        else:
            job_output = self.work_dir / f"train_{job_id}.out"
        print(f"Job output is in {job_output}")

        if self.task_report_id is not None:
            try:
                with open(job_output) as job_output_file:
                    self.slurm_output = job_output_file.read()
            except FileNotFoundError:
                print(f"File {job_output} not found.")
                self.slurm_output = ""

            self.job_stats = slurm_job_stats(job_id)
            
            # get current task report so we can add to messages
            task_report = self.api.task_details(self.task_report_id)
            self.api.task_update(
                self.task_report_id,
                task_report.label,
                task_report.user,
                f"{task_report.messages}\n Slurm job output:\n{self.slurm_output}\n\n{self.job_stats}\n{'='*80}",
            )

        # when cancelled via delete button on mydella web ui,
        # statuses are COMPLETED,CANCELLED
        # if time limit ran out, status will include TIMEOUT as well as CANCELLED
        if "CANCELLED" in job_status and "TIMEOUT" not in job_status:
            raise JobCancelled

    def segmentation_training(self):
        # get absolute versions of these paths _before_ changing working directory
        abs_training_data_dir = self.training_data_dir.absolute()
        abs_model_file = self.model_file.absolute()
        abs_output_modelfile = self.output_modelfile.absolute()

        # change directory to working directory, since by default,
        # slurm executes the job from the directory where it was submitted
        os.chdir(self.work_dir)
        
        training_data_size = sum(f.stat().st_size for f in abs_training_data_dir.glob('*[!.xml]') if f.is_file())
        print(f"Training data size: {training_data_size}")
        
        prelim_cpu_mem = estimate_cpu_mem(training_data_size, self.training_mode)
        prelim_train_time = estimate_duration(training_data_size, self.training_mode)
        
        print(f"Requesting {prelim_cpu_mem} at {prelim_train_time}.")

        job_id = segtrain(
            abs_training_data_dir,
            abs_model_file,
            abs_output_modelfile,
            self.num_workers,
            mem_per_cpu = prelim_cpu_mem,
            training_time = prelim_train_time,
        )
        # change back to original working directory
        os.chdir(self.orig_working_dir)
        self.monitor_slurm_job(job_id)
        
        # need to check if there is a _best.mlmodel
        prelim_best = list(self.output_model_dir.glob("*_best.mlmodel"))
        if prelim_best:
            self.upload_best()
            print("Best model already found.")
            return
        
        # check exit status first
        job_status = slurm_job_status(job_id)
        if "OUT_OF_MEMORY" in job_status:
            # might want to split up handling here more. OUT_OF_MEMORY might indicate
            # odd cases like a seg train task with no regions, or it might indicate
            # that the memory should be raised and the train task attempted again.
            self.upload_best()
            
        else:
        
            # find preliminary model with highest accuracy to use as input for next train job
            best_epoch_acc = slurm_get_max_acc(self.slurm_output, self.job_stats)
            if best_epoch_acc:
                prelim_best_model = list(self.output_model_dir.glob(f"*_{best_epoch_acc[0]}.mlmodel"))[0]
                prelim_model_file = get_prelim_model(prelim_best_model)
                print(f"Preliminary best model: {prelim_model_file}.")
        
            epoch_max_acc, max_acc = slurm_get_max_acc(self.slurm_output, self.training_mode)
            full_duration = calc_full_duration(self.slurm_output, self.job_stats)
            mem_per_cpu = calc_cpu_mem(self.job_stats)
            
            print(f"""The recommended mem per cpu is {mem_per_cpu}.
            The recommended duration time is {full_duration}.
            The epoch with the highest accuracy was {epoch_max_acc} with {max_acc}.""")

        if self.update:
            self.upload_best()
        else:
            self.upload_all_models()

    def recognition_training(self):
        # NOTE: this is nearly the same as segmentation_training method

        # get absolute versions of these paths _before_ changing working directory
        abs_training_data_dir = self.training_data_dir.absolute()
        # input model is optional
        abs_model_file = self.model_file.absolute() if self.model_file else None
        abs_output_modelfile = self.output_modelfile.absolute()

        # change directory to working directory, since by default,
        # slurm executes the job from the directory where it was submitted
        os.chdir(self.work_dir)
        
        training_data_file = abs_training_data_dir / "train.arrow"
        training_data_size = training_data_file.stat().st_size
        print(f"Training data size: {training_data_size}")
        
        prelim_cpu_mem = estimate_cpu_mem(training_data_size, self.training_mode)
        prelim_train_time = estimate_duration(training_data_size, self.training_mode)
        
        print(f"Requesting {prelim_cpu_mem} at {prelim_train_time}.")

        job_id = recognition_train(
            abs_training_data_dir,
            abs_output_modelfile,
            abs_model_file,
            self.num_workers,
            mem_per_cpu = prelim_cpu_mem,
            training_time = prelim_train_time,
        )
        # change back to original working directory
        os.chdir(self.orig_working_dir)
        self.monitor_slurm_job(job_id)
        
        # need to check if there is a _best.mlmodel
        prelim_best = list(self.output_model_dir.glob("*_best.mlmodel"))
        if prelim_best:
            self.upload_best()
            print("Best model already found.")
            return
        
        # check exit status first
        job_status = slurm_job_status(job_id)
        if "OUT_OF_MEMORY" in job_status:
            # might want to split up handling here more. OUT_OF_MEMORY might indicate
            # odd cases like a seg train task with no regions, or it might indicate
            # that the memory should be raised and the train task attempted again.
            self.upload_best()
            
        else:
        
            # find preliminary model with highest accuracy to use as input for next train job
            best_epoch_acc = slurm_get_max_acc(self.slurm_output, self.job_stats)
            if best_epoch_acc:
                prelim_best_model = list(self.output_model_dir.glob(f"*_{best_epoch_acc[0]}.mlmodel"))[0]
                prelim_model_file = get_prelim_model(prelim_best_model)
                print(f"Preliminary best model: {prelim_model_file}.")
        
            epoch_max_acc, max_acc = slurm_get_max_acc(self.slurm_output, self.training_mode)
            full_duration = calc_full_duration(self.slurm_output, self.job_stats)
            mem_per_cpu = calc_cpu_mem(self.job_stats)
            
            print(f"""The recommended mem per cpu is {mem_per_cpu}.
            The recommended duration time is {full_duration}.
            The epoch with the highest accuracy was {epoch_max_acc} with {max_acc}.""")
            
            self.upload_best()

    def upload_best(self):
        # look for and upload best model

        # when update is requested, specify model id to be updated
        if self.update:
            model_id = self.model_id
        else:
            model_id = None

        # in certain cases we only want to upload the model to
        # eScriptorium if it has improved on the original model;
        # pass in original model for minimum accuracy comparison
        # when update mode is update-if-improved
        compare_model_file = None
        if self.update == UpdateMode.IF_IMPROVED and self.model_file:
            compare_model_file = self.model_file.absolute()

        best_model = upload_best_model(
            self.api,
            self.output_modelfile.parent,
            self.training_mode,
            model_id=model_id,
            original_model=compare_model_file,
        )
        if best_model:
            # TODO: revise message to include info about created/updated model id ##
            print(f"Uploaded {best_model} to eScriptorum")
        else:
            # possibly best model found but upload failed?
            print("No best model found")

    def upload_all_models(self):
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
    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument(
        "-u",
        "--update",
        help="Update the specified model with the best model from training (requires --model)",
        dest="update",
        default=UpdateMode.NEVER,
        action="store_const",
        const=UpdateMode.ALWAYS,
        required=False,
    )
    update_group.add_argument(
        "--update-if-improved",
        help="Update the specified model with the best model from training ONLY if improved on original",
        dest="update",
        action="store_const",
        const=UpdateMode.IF_IMPROVED,
        required=False,
    )
    parser.add_argument(
        "--model-name",
        help="Name to be used for newly trained model (not compatible with --update)",
        type=str,
        dest="model_name",
        required=False,
    )
    parser.add_argument(
        "-p",
        "--parts",
        help="Optional list of part ids for training. Format as #,#,#  or #-##."
        + "(if not specified, uses entire document)",
        type=intspan,
    )
    parser.add_argument(
        "-tr",
        "--task-report",
        help="Optional task report id, for reporting sbatch and slurm output",
        type=int,
        dest="task_report_id",
        required=False,
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
    # validate argument combinations

    # when update or update-if-modified is specified, model is required
    if args.update:
        error_messages = []
        if not args.model_id:
            error_messages.append("cannot use --update without specifying --model")
        if args.model_name:
            error_messages.append("cannot specify both --model-name and --update")
        if error_messages:
            print(f"Error: {'; '.join(error_messages)}")
            sys.exit(1)
    if not any([args.model_id, args.model_name]):
        print(f"Error: one of --model or --model-name is required")
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
    logger_local = logging.getLogger("htr2hpc")
    logger_local.setLevel(logging.INFO)
    # output kraken logging details to confirm binary data looks ok
    logger_kraken = logging.getLogger("kraken")
    # logger_kraken.setLevel(logging.INFO)

    # nearly all the argparse options need to be passed to the training manager class
    # convert to a _copy_ dictionary and delete the unused parmeters
    arg_options = dict(vars(args))
    del arg_options["clean"]
    del arg_options["mode"]  # converted to training_mode (Segment/Recognize)

    # initialize training manager
    try:
        training_mgr = TrainingManager(
            api_token=api_token, training_mode=es_model_jobs[args.mode], **arg_options
        )
    except ConnectionError as err:
        print(err)
        print(
            "Check that you have specified the correct BASE_URL and API token and confirm the eScriptorium server is available."
        )
        sys.exit(1)

    try:
        # prep data for training
        training_mgr.training_prep()
        # run training for requested mode
        if args.mode == "segmentation":
            training_mgr.segmentation_training()
        if args.mode == "transcription":
            training_mgr.recognition_training()
    except (NotFound, NotAllowed) as err:
        print(f"Something went wrong: {err}")
    except JobCancelled as err:
        print(f"Slurm job was cancelled")

    # unless requested not to, clean up the working directory, which includes:
    # - downloaded training data & model to fine tune
    # - generated models
    # - training output
    if args.clean:
        print(
            f"Removing working directory {args.work_dir} with all training data and models."
        )
        rmtree(args.work_dir)


if __name__ == "__main__":
    main()
