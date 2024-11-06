#!/usr/bin/env python
import argparse
import os
import sys

import parsl
from parsl.configs.local_threads import config


from htr2hpc.train_apps import prep_training_data, segtrain


api_token_env_var = "ESCRIPTORIUM_API_TOKEN"

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
    parser.add_argument(
        "-t", "--transcription", help="Transcription id to export", type=int
    )
    args = parser.parse_args()

    parsl.load(config)
    training_data = prep_training_data(
        args.base_url,
        api_token,
        args.document_id,
        args.transcription,
    )
    # NOTE: print currently displays stdout with training progress
    # TODO: need to pass in an output file for the bash app to return best model (?)
    print(segtrain(inputs=[training_data]).result())

    # example run on my local dev:
    # ./src/htr2hpc/train.py http://localhost:8000/ -u 2 -d 1 -n "test doc" -t 1
