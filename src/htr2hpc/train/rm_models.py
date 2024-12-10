"""
Utility script to delete models from eScriptorim API, for easily
cleaning up models created and uploaded for testing.

Requires an eScriptorium API token set as an environment variable in
**ESCRIPTORIUM_API_TOKEN**.

Takes a base url for the eScriptorium instance and a model name prefix; will
delete all models that start with the specified model name prefix.

usage:

    python src/htr2hpc/train/rm_models.py https://test-htr.lib.princeton.edu/ model_prefix

"""

import os
import argparse
import sys

from tqdm import tqdm

from htr2hpc.api_client import eScriptoriumAPIClient

api_token_env_var = "ESCRIPTORIUM_API_TOKEN"


def main():
    try:
        api_token = os.environ[api_token_env_var]
    except KeyError:
        print(
            f"Error: eScriptorium API token must be set as environment variable {api_token_env_var}",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Remove models from eScriptorium API")
    parser.add_argument(
        "base_url",
        metavar="BASE_URL",
        help="Base URL for eScriptorium instance (without /api/)",
        type=str,
    )

    parser.add_argument(
        "model_name",
        help="Model name to remove",
        type=str,
    )
    args = parser.parse_args()

    api = eScriptoriumAPIClient(args.base_url, api_token=api_token)
    model_list = api.model_list()
    rm_models = []
    # filter by match on name
    while model_list.next:
        rm_models.extend(
            [
                model
                for model in model_list.results
                if model.name.startswith(args.model_name)
            ]
        )
        model_list = model_list.next_page()

    if rm_models:
        print(f"Removing {len(rm_models)} models")
        for model in tqdm(rm_models, desc="Removing models: "):
            api.model_delete(model.pk)
    else:
        print("No matching models")


if __name__ == "__main__":
    main()
