#!/usr/bin/env python
import argparse
import os
import sys
import logging
import pathlib
from multiprocessing import cpu_count

import parsl
from kraken.kraken import SEGMENTATION_DEFAULT_MODEL, DEFAULT_MODEL

from htr2hpc.api_client import eScriptoriumAPIClient
from htr2hpc.train.apps import prep_training_data, segtrain, get_model
from htr2hpc.train.config import parsl_config


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
        title="mode", description="supported training modes", required=True
    )
    segmentation_parser = subparsers.add_parser("segmentation")
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
    parser.add_argument(
        "-p",
        "--parts",
        help="Optional list of parts to train on (if not entire document)",
        # TODO: use list or intspan here ?
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
    )
    args = parser.parse_args()

    # make sure working directory does not already exist
    # TODO: allow using an existing dir+data, or is that only a dev issue?
    if args.work_dir.exists():
        print(f"Working directory `{args.work_dir}` already exists", file=sys.stderr)
        sys.exit(1)
    # create new working directory
    args.work_dir.mkdir()

    logging.basicConfig(encoding="utf-8", level=logging.WARN)
    logger_upscope = logging.getLogger("htr2hpc")
    logger_upscope.setLevel(logging.INFO)

    parsl.load(parsl_config)

    api = eScriptoriumAPIClient(args.base_url, api_token=api_token)

    # TODO : check api access works before going too far?
    # (currently does not handle connection error gracefully)

    training_data_dir = prep_training_data(
        api,
        args.work_dir,
        args.document_id,
        # TODO: optional part ids
    )
    # if there is an error getting the training data, we get a
    # parsl.dataflow.errors.DependencyError

    # if model id is specified, download the model from escriptorium API,
    # confirming that it is the appropriate type (segmentation/transcription)
    if args.model_id:
        model_file = get_model(
            api,
            args.model_id,
            es_model_jobs[args.job],
            args.work_dir,
        )

    # if model id is not specified, use the default from kraken
    else:
        # get the appropriate model file for the requested training mode
        # kraken default defs are path objects
        model_file = default_model[args.job]

    # TODO: segtrain/ train should be wrapped by a join app,
    # so that once training completes we can determine whether or not to
    # upload a new model to eScriptorium

    # TODO: determine which job to run based on the input
    # - get input data for that job
    # - run the bash app with the slurm provider
    # - get the result from the bash app and check for failure/success
    #

    # except parsl.dataflow.errors.DependencyError as err:
    # if there is an error getting the training data, we get a
    # parsl.dataflow.errors.DependencyError

    # create a directory and path for the output model file
    output_model_dir = args.work_dir / "output_model"
    output_model_dir.mkdir()
    output_modelfile = output_model_dir / "model"

    # get absolute versions of these paths _before_ changing working directory
    abs_training_data_dir = training_data_dir.absolute()
    abs_model_file = model_file.absolute()
    abs_output_modelfile = output_modelfile.absolute()

    # change directory to working directory
    # by default, slurm executes the job from the directory where it was submitted
    # this is where parsl will put the sbatch file also
    os.chdir(args.work_dir)

    if args.job == "segmentation":
        try:
            print(
                segtrain(
                    inputs=[
                        abs_training_data_dir,
                        abs_model_file,
                        abs_output_modelfile,
                        args.workers,
                    ]
                ).result()
            )
        except parsl.app.errors.BashExitFailure as err:
            print(f"Something went wrong: {err}")

    # TODO: handle transcription training

    # example to run against local dev instance:
    # setenv ESCRIPTORIUM_API_TOKEN "####"
    # ./src/htr2hpc/train.py transcription http://localhost:8000/ test_doc1 -d 1 -t 1

    # when this is all working, cleanup working dir (by default, with option to skip)

    #  cleanup
    parsl.dfk().cleanup()


if __name__ == "__main__":
    main()
