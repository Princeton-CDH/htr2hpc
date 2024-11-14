#!/usr/bin/env python
import argparse
import os
import sys
import logging
import pathlib
from multiprocessing import cpu_count

import parsl

from htr2hpc.api_client import eScriptoriumAPIClient
from htr2hpc.train.apps import prep_training_data, segtrain, get_model
from htr2hpc.train.config import parsl_config


api_token_env_var = "ESCRIPTORIUM_API_TOKEN"

# map our job type option choices to escriptorium terms
es_model_jobs = {"segmentation": "Segment", "transcription": "Recognize"}


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
    # - working directory
    # - create/update model flag (make explicit)
    #   name for the new model when creating a new one (required)

    parser = argparse.ArgumentParser(
        description="Export content from eScriptorium and train or fine-tune models"
    )
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
        "-d", "--document", help="Document id to export", type=int, dest="document_id"
    )
    # parser.add_argument(
    # "-t", "--transcription", help="Transcription id to export", type=int
    # )
    # transcription id matters for recognition training since a document may have multiple
    # - part export logic may be different to get the correct transcription text
    parser.add_argument(
        "-j",
        "--job",
        help="Job type (segmentation or transcription)",
        type=str,
        choices=["segmentation", "transcription"],
        default="segmentation",  # for convenience, for now
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
        # TODO: use list or intspan here
    )
    parser.add_argument(
        "-w",
        "--workers",
        help="Number of workers for training task (default: %(default)d",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    # make sure working directory does not already exist
    if args.work_dir.exists():
        print(f"Working directory `{args.work_dir}` already exists", file=sys.stderr)
        sys.exit(1)
    # create new working directory
    args.work_dir.mkdir()

    # logging.basicConfig(filename=sys.stdout, encoding="utf-8", level=logging.DEBUG)
    logging.basicConfig(encoding="utf-8", level=logging.WARN)
    logger_upscope = logging.getLogger("htr2hpc")
    logger_upscope.setLevel(logging.INFO)

    parsl.load(parsl_config)

    api = eScriptoriumAPIClient(args.base_url, api_token=api_token)

    # TODO : check api access works before going too far?

    training_data_dir = prep_training_data(
        api,
        args.work_dir,
        args.document_id,
        # TODO: optional part ids
    )
    # if there is an error getting the training data, we get a
    # parsl.dataflow.errors.DependencyError

    if args.model_id:
        model_file = get_model(
            api,
            args.model_id,
            es_model_jobs[args.job],
            args.work_dir,
        )
    # TODO: if not specified, get default model for training mode

    # NOTE: print currently displays stdout with training progress
    # TODO: need to pass in an output file for the bash app to return best model (?)

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

    # change directory to working directory
    # by default, slurm executes the job from the directory where it was submitted
    os.chdir(args.work_dir)

    if args.job == "segmentation":
        try:
            print(
                segtrain(
                    inputs=[
                        # use absolute paths
                        training_data_dir.absolute(),
                        model_file.absolute(),
                        output_modelfile.absolute(),
                        args.workers,
                    ]
                ).result()
            )
        except parsl.app.errors.BashExitFailure as err:
            print(f"Something went wrong: {err}")

    # TODO: handle transcription training

    # example to run against local dev instance:
    # setenv ESCRIPTORIUM_API_TOKEN "####"
    # ./src/htr2hpc/train.py http://localhost:8000/ -d 1 -t 1

    # when this is all working, cleanup working dir (by default, option to skip)

    # parsl cleanup
    parsl.dfk().cleanup()


if __name__ == "__main__":
    main()
