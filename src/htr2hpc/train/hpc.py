"""Utilities for managing the remote HPC conda environment."""
import logging

from htr2hpc import __version__

logger = logging.getLogger(__name__)

ANACONDA_MODULE = "anaconda3/2025.6"


def ensure_htr2hpc_version(conn):
    """Install the currently deployed version of htr2hpc in the remote conda
    env, ensuring htr2hpc and all its dependencies (including kraken) match
    the deployed version. If the correct version is already installed, pip
    does nothing."""
    install_cmd = (
        f"module load {ANACONDA_MODULE} && "
        "conda run -n htr2hpc pip install -q "
        f"git+https://github.com/Princeton-CDH/htr2hpc.git@v{__version__}#egg=htr2hpc"
    )
    result = conn.run(install_cmd, warn=True, hide=True)
    if result.exited != 0:
        logger.warning(
            f"Could not install htr2hpc v{__version__} in conda env: {result.stderr}"
        )
        return False
    logger.info(f"htr2hpc=={__version__} is up to date in conda env")
    return True
