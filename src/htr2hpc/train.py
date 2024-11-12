#!/usr/bin/env python
import argparse
import os
import sys
import logging
import pathlib
from multiprocessing import cpu_count

import parsl
from parsl.configs.local_threads import config

from htr2hpc.api_client import eScriptoriumAPIClient
from htr2hpc.train_apps import (
    prep_training_data,
    segtrain,
    get_segmentation_data,
    get_model,
)


api_token_env_var = "ESCRIPTORIUM_API_TOKEN"

# map our job type option choices to escriptorium terms
es_model_jobs = {"segmentation": "Segment", "transcription": "Recognize"}

if __name__ == "__main__":
    try:
        api_token = os.environ[api_token_env_var]
    except KeyError:
        print(
            f"Error: eScriptorium API token must be set as environment variable {api_token_env_var}",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Export content from eScriptorium and train or fine-tune models"
    )
    parser.add_argument(
        "base_url",
        help="Base URL for eScriptorium instance (do not include /api/)",
        type=str,
    )
    parser.add_argument(
        "-d", "--document", help="Document id to export", type=int, dest="document_id"
    )
    # parser.add_argument(
    # "-t", "--transcription", help="Transcription id to export", type=int
    # )
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
        help="Number of workers for training task (default: %(default)s)",
        default=cpu_count(),
    )
    args = parser.parse_args()

    # logging.basicConfig(filename=sys.stdout, encoding="utf-8", level=logging.DEBUG)
    logging.basicConfig(encoding="utf-8", level=logging.WARN)
    logger_upscope = logging.getLogger("htr2hpc")
    logger_upscope.setLevel(logging.DEBUG)

    parsl.load(config)

    api = eScriptoriumAPIClient(args.base_url, api_token=api_token)

    training_data_dir = prep_training_data(
        api,
        args.document_id,
        # TODO: optional part ids
    )
    # if there is an error getting the training data, we get a
    # parsl.dataflow.errors.DependencyError

    if args.model_id:
        model_file = get_model(
            api, args.model_id, es_model_jobs[args.job], training_data_dir
        )
    # TODO: if not specified, get default model for mode

    # NOTE: print currently displays stdout with training progress
    # TODO: need to pass in an output file for the bash app to return best model (?)

    if args.job == "segmentation":
        try:
            print(
                segtrain(inputs=[training_data_dir, model_file, args.workers]).result()
            )
        except parsl.dataflow.errors.DependencyError as err:
            # if there is an error getting the training data, we get a
            # parsl.dataflow.errors.DependencyError
            print(f"Something went wrong: {err}")

    # TODO: handle transcription training

    # example to run against local dev instance:
    # setenv ESCRIPTORIUM_API_TOKEN "####"
    # ./src/htr2hpc/train.py http://localhost:8000/ -d 1 -t 1
