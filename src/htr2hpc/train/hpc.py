"""Utilities for managing the remote HPC conda environment."""
import logging

from django.conf import settings

logger = logging.getLogger(__name__)


def ensure_htr2hpc_version(conn):
    """Install the currently deployed version of htr2hpc in the remote conda
    env, ensuring htr2hpc and all its dependencies (including kraken) match
    the deployed version. Uses --upgrade so that pip re-evaluates dependencies
    even when htr2hpc itself is already at the correct version.

    Uses HTR2HPC_GITREF (set by Ansible at deploy time) so that HPC always
    installs the same ref that is running on the web server — whether that is
    a release tag or a development branch."""
    install_cmd = (
        f"module load {settings.HPC_ANACONDA_MODULE} && "
        "conda run -n htr2hpc pip install -q --upgrade "
        f"git+https://github.com/Princeton-CDH/htr2hpc.git@{settings.HTR2HPC_GITREF}#egg=htr2hpc"
    )
    result = conn.run(install_cmd, warn=True, hide=True)
    if result.exited != 0:
        logger.warning(
            f"Could not install htr2hpc {settings.HTR2HPC_GITREF} in conda env: {result.stderr}"
        )
        return False
    logger.info(f"htr2hpc {settings.HTR2HPC_GITREF} is up to date in conda env")
    return True
