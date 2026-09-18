"""Utilities for managing the remote HPC conda environment."""

import logging

from django.conf import settings

from htr2hpc import __version__

logger = logging.getLogger(__name__)


def ensure_htr2hpc_version(conn):
    """Install the currently deployed version of htr2hpc in the remote conda
    env. Uses --force-reinstall --no-deps so that pip always reinstalls
    htr2hpc itself when the gitref changes, even if the version number has not
    changed (e.g. two different commits at the same 0.x.dev0 version).
    --no-deps avoids reinstalling large dependencies (torch, kraken, etc.)
    that are already correctly installed in the conda env.

    Uses HTR2HPC_GITREF when set (staging deploys: exact commit SHA set by
    Ansible), otherwise falls back to the current version tag."""
    gitref = getattr(settings, "HTR2HPC_GITREF", __version__)
    # NOTE: when installing by version, the version number must match a git tag exactly
    # TODO: when htr2hpc is later switched to publish on PyPI, production should use
    # pip install htr2hpc=={version} and staging should keep the git+SHA URL.
    install_cmd = (
        f"module load {settings.HPC_ANACONDA_MODULE} && "
        "conda run -n htr2hpc pip install --force-reinstall --no-deps "
        f"git+https://github.com/Princeton-CDH/htr2hpc.git@{gitref}#egg=htr2hpc"
    )
    result = conn.run(install_cmd, warn=True, hide=True)
    if result.exited != 0:
        logger.warning(
            f"Could not install htr2hpc {gitref} in conda env: {result.stderr}"
        )
        return False
    logger.info(f"htr2hpc {gitref} is up to date in conda env")
    return True
