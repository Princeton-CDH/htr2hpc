"""Utilities for managing the remote HPC conda environment."""

import logging
from pathlib import Path

from django.conf import settings
from filelock import FileLock, Timeout

from htr2hpc import __version__

logger = logging.getLogger(__name__)


def ensure_htr2hpc_version(conn):
    """Install the currently deployed version of htr2hpc in the remote conda
    env, ensuring htr2hpc and all its dependencies (including kraken) match
    the deployed version. Uses --force-reinstall so that pip always reinstalls
    when the gitref changes, even if the version number has not changed (e.g.
    two different commits at the same 0.x.dev0 version).

    Uses HTR2HPC_GITREF when set (staging deploys: exact commit SHA set by
    Ansible), otherwise falls back to the current version tag.

    Uses a per-user file lock on shared NFS storage to coordinate concurrent
    installs across all app hosts. If the lock is already held by another job,
    that job is already running the install — wait for it to finish and skip
    the install. If no lock is held, acquire it and run the install.

    Skips the install entirely if the user has active Slurm jobs, since
    reinstalling packages while training is running would corrupt the conda
    environment mid-training."""
    gitref = getattr(settings, "HTR2HPC_GITREF", __version__)
    # NOTE: when installing by version, the version number must match a git tag exactly
    # TODO: when htr2hpc is later switched to publish on PyPI, production should use
    # pip install htr2hpc=={version} and staging should keep the git+SHA URL.
    install_cmd = (
        f"module load {settings.HPC_ANACONDA_MODULE} && "
        "flock -w 300 ~/.htr2hpc-conda-install.lock "
        "conda run -n htr2hpc pip install --force-reinstall "
        f"git+https://github.com/Princeton-CDH/htr2hpc.git@{gitref}#egg=htr2hpc"
    )
    lock_path = Path(settings.MEDIA_ROOT) / f"htr2hpc-conda-install-{conn.user}.lock"
    lock = FileLock(lock_path)

    # Skip install if the user has active htr2hpc Slurm jobs — reinstalling
    # packages while training is running would corrupt the conda environment
    # mid-training. Only check for htr2hpc job names (segtrain:/train:/
    # calibrate_segtrain:/calibrate_train:) to avoid false positives from
    # unrelated jobs the user may have running on the cluster.
    squeue_result = conn.run(f"squeue -u {conn.user} -h -o '%j'", warn=True, hide=True)
    if squeue_result.exited != 0:
        logger.warning(
            f"squeue check failed for {conn.user}, skipping install to be safe"
        )
        return True
    active_htr_jobs = [
        line
        for line in squeue_result.stdout.strip().splitlines()
        if line.startswith(
            ("segtrain:", "train:", "calibrate_segtrain:", "calibrate_train:")
        )
    ]
    if active_htr_jobs:
        logger.info(f"User {conn.user} has active htr2hpc Slurm jobs, skipping install")
        return True

    # Try to acquire the lock immediately (non-blocking).
    # If we get it, we are the first job — run the install.
    # If we don't get it, another job is already installing — wait for it to
    # finish and skip the install.
    try:
        lock.acquire(timeout=0)
    except Timeout:
        logger.info(f"Another job is installing htr2hpc for {conn.user}, waiting...")
        try:
            with FileLock(lock_path, timeout=900):
                pass  # wait for the other job to finish
        except Timeout:
            logger.warning(
                f"Timed out waiting for htr2hpc install to complete for {conn.user}"
            )
            return False
        logger.info(
            f"htr2hpc install completed by another job, skipping for {conn.user}"
        )
        return True

    # We hold the lock — run the install
    try:
        result = conn.run(install_cmd, warn=True, hide=True)
    finally:
        lock.release()

    if result.exited != 0:
        logger.warning(
            f"Could not install htr2hpc {gitref} in conda env: {result.stderr}"
        )
        return False
    logger.info(f"htr2hpc {gitref} is up to date in conda env")
    return True
