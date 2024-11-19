#!/usr/bin/env python
import argparse
import os
import sys
import logging
import pathlib
from time import sleep

from kraken.kraken import SEGMENTATION_DEFAULT_MODEL, DEFAULT_MODEL
from tqdm import tqdm

from htr2hpc.api_client import eScriptoriumAPIClient
from htr2hpc.train.apps import (
    get_training_data,
    segtrain,
    get_model,
    slurm_job_status,
    slurm_job_queue_status,
)


api_token_env_var = "ESCRIPTORIUM_API_TOKEN"

# map our job type option choices to escriptorium terms
es_model_jobs = {"segmentation": "Segment", "transcription": "Recognize"}

# kraken python package provides paths to the default best models for both modes
default_model = {
    "segmentation": SEGMENTATION_DEFAULT_MODEL,
    "transcription": DEFAULT_MODEL,
}


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
        help="Working directory where data should be downloaded (must already exist)",
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

    # not supported yet, handle later
    # parser.add_argument(
    #     "-p",
    #     "--parts",
    #     help="Optional list of parts to train on (if not entire document)",
    #     # TODO: use list or intspan here ?
    # )
    parser.add_argument(
        "--existing-data",
        help="Use existing data from a previous run",
        action="store_true",
        default=False,
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

    # NOTE: needs to match the number in the parsl config...
    # how best to configure this?
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
    # TODO: allow using an existing dir+data, or is that only a dev issue?
    if args.work_dir.exists() and not args.existing_data:
        print(
            f"Working directory `{args.work_dir}` already exists (use --existing-data to allow)",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.existing_data and not args.work_dir.exists():
        print(
            f"Specified working directory `{args.work_dir}` already exists (use --existing-data allow)",
            file=sys.stderr,
        )
        sys.exit(1)

    # create new working directory if it doesn't already exist
    if not args.existing_data:
        args.work_dir.mkdir()

    logging.basicConfig(encoding="utf-8", level=logging.WARN)
    logger_upscope = logging.getLogger("htr2hpc")
    logger_upscope.setLevel(logging.DEBUG)

    api = eScriptoriumAPIClient(args.base_url, api_token=api_token)

    # TODO : check api access works before going too far?
    # (currently does not handle connection error gracefully)

    training_data_dir = args.work_dir / "parts"
    if not args.existing_data:
        training_data_dir.mkdir()
        get_training_data(
            api,
            training_data_dir,
            args.document_id,
            # TODO: optional part ids
        )

    # if model id is specified, download the model from escriptorium API,
    # confirming that it is the appropriate type (segmentation/transcription)
    # NOTE: currently ignores existing data flag, since we need model file name
    if args.model_id:
        model_file = get_model(
            api,
            args.model_id,
            es_model_jobs[args.mode],
            args.work_dir,
        )

    # if model id is not specified, use the default from kraken
    else:
        # get the appropriate model file for the requested training mode
        # kraken default defs are path objects
        model_file = default_model[args.mode]

    # TODO: determine which job to run based on the input
    # - get input data for that job
    # - run the bash app with the slurm provider
    # - get the result from the bash app and check for failure/success
    #

    # create a directory and path for the output model file
    output_model_dir = args.work_dir / "output_model"
    # currently assuming model dir is empty
    if args.existing_data:
        output_model_dir.rmdir()
    output_model_dir.mkdir()  # TODO: allow to exist?
    output_modelfile = output_model_dir / "model"

    # get absolute versions of these paths _before_ changing working directory
    abs_training_data_dir = training_data_dir.absolute()
    abs_model_file = model_file.absolute()
    abs_output_modelfile = output_modelfile.absolute()

    # change directory to working directory, since by default,
    # slurm executes the job from the directory where it was submitted
    os.chdir(args.work_dir)

    if args.mode == "segmentation":
        job_id = segtrain(
            abs_training_data_dir,
            abs_model_file,
            abs_output_modelfile,
            args.num_workers,
        )
        # get job status (presumably PENDING)
        job_status = slurm_job_queue_status(job_id)
        # typical states are PENDING, RUNNING, SUSPENDED, COMPLETING, and COMPLETED.
        # https://slurm.schedmd.com/job_state_codes.html
        # end states could be FAILED, CANCELLED, OUT_OF_MEMORY, TIMEOUT
        # * but note that squeue only reports on pending & running jobs

        # loop while the job is pending or running and then stop
        # use tqdm to display job status and wait time
        with tqdm(
            desc=f"Slurm job {job_id}",
            bar_format="{desc}{postfix}    | {elapsed}",
        ) as statusbar:
            while job_status:
                statusbar.set_postfix_str(f"status: {job_status}")
                sleep(3)
                job_status = slurm_job_queue_status(job_id)

        # check the completed status
        job_status = slurm_job_status(job_id)
        print(
            f"Job {job_id} is no longer queued; ending status: {','.join(job_status)}"
        )
        job_output = training_data_dir / f"segtrain_{job_id}.out"
        print(f"Job output should be in {job_output}")

        # TODO: if it completed (or timeout?), check for results
        # - if model improved, upload to eScriptorium as new or updated model

    # TODO: handle transcription training

    #  TODO cleanup
    # when this is all working, cleanup working dir (by default, with option to skip)


if __name__ == "__main__":
    main()
