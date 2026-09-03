"""Utilities for managing the remote HPC conda environment."""
import logging

from django.conf import settings

from htr2hpc import __version__

logger = logging.getLogger(__name__)


def ensure_htr2hpc_version(conn):
    """Install the currently deployed version of htr2hpc in the remote conda
    env, ensuring htr2hpc and all its dependencies (including kraken) match
    the deployed version. Uses --upgrade so that pip re-evaluates dependencies
    even when htr2hpc itself is already at the correct version.

    Uses HTR2HPC_GITREF when set (staging deploys: exact commit SHA set by
    Ansible), otherwise falls back to the current version tag."""
    gitref = getattr(settings, "HTR2HPC_GITREF", __version__)
    install_cmd = (
        f"module load {settings.HPC_ANACONDA_MODULE} && "
        "conda run -n htr2hpc pip install -q --upgrade "
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
