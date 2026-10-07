"""Utilities for managing the remote HPC conda environment."""

import logging
import time

from django.conf import settings

from htr2hpc import __version__

logger = logging.getLogger(__name__)

DOTFILE_PATH = "~/.htr2hpc_version"
POLL_INTERVAL = 5  # seconds between polls when status is "installing"
POLL_TIMEOUT = 900  # max seconds to wait for another install to complete


def _read_dotfile(conn):
    """Read and parse the version dotfile from Adroit.
    Returns a dict with keys 'gitref', 'version', 'status', or None if not found."""
    result = conn.run(f"cat {DOTFILE_PATH}", warn=True, hide=True)
    if result.exited != 0:
        return None
    data = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            data[key.strip()] = value.strip()
    return data


def _write_dotfile(conn, status, gitref="", version=""):
    """Write the version dotfile to Adroit."""
    content = f"gitref={gitref}\\nversion={version}\\nstatus={status}"
    conn.run(f"printf '{content}' > {DOTFILE_PATH}", warn=True, hide=True)


def _delete_dotfile(conn):
    """Delete the version dotfile from Adroit."""
    conn.run(f"rm -f {DOTFILE_PATH}", warn=True, hide=True)


def _version_matches(dotfile, desired_gitref, desired_version):
    """Check if the dotfile records the desired version.
    Uses gitref for staging (when HTR2HPC_GITREF is set), version for production."""
    if desired_gitref:
        return dotfile.get("gitref") == desired_gitref
    return dotfile.get("version") == desired_version


def ensure_htr2hpc_version(conn):
    """Install the currently deployed version of htr2hpc in the remote conda
    env, ensuring htr2hpc and all its dependencies (including kraken) match
    the deployed version. Uses --force-reinstall so that pip always reinstalls
    when the gitref changes, even if the version number has not changed (e.g.
    two different commits at the same 0.x.dev0 version).

    Uses HTR2HPC_GITREF when set (staging deploys: exact commit SHA set by
    Ansible), otherwise falls back to the current version tag.

    Uses a dotfile (~/.htr2hpc_version) on the HPC host to track the installed
    version and coordinate concurrent installs. If the dotfile shows the correct
    version is already installed, the install is skipped. If another job is
    currently installing, this job waits for it to finish."""
    desired_gitref = getattr(settings, "HTR2HPC_GITREF", "")
    desired_version = __version__
    install_cmd = (
        f"module load {settings.HPC_ANACONDA_MODULE} && "
        "conda run -n htr2hpc pip install --force-reinstall "
        f"git+https://github.com/Princeton-CDH/htr2hpc.git@{desired_gitref or desired_version}#egg=htr2hpc"
    )

    dotfile = _read_dotfile(conn)

    if dotfile is None:
        # No dotfile — we are the first job, run the install
        _write_dotfile(conn, status="installing")
        result = conn.run(install_cmd, warn=True, hide=True)
        if result.exited != 0:
            logger.warning(
                f"Could not install htr2hpc in conda env for {conn.user}: {result.stderr}"
            )
            _delete_dotfile(conn)
            return False
        _write_dotfile(
            conn, status="installed", gitref=desired_gitref, version=desired_version
        )
        logger.info(
            f"htr2hpc {desired_gitref or desired_version} installed for {conn.user}"
        )
        return True

    if dotfile.get("status") == "installing":
        # Another job is installing — poll until it finishes
        logger.info(f"Another job is installing htr2hpc for {conn.user}, waiting...")
        elapsed = 0
        while elapsed < POLL_TIMEOUT:
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            dotfile = _read_dotfile(conn)
            if dotfile is None or dotfile.get("status") != "installing":
                break
        else:
            logger.warning(
                f"Timed out waiting for htr2hpc install to complete for {conn.user}"
            )
            return False
        # Re-read after waiting; fall through to version check below
        dotfile = _read_dotfile(conn)
        if dotfile is None:
            # Dotfile was deleted (install failed) — try installing ourselves
            _write_dotfile(conn, status="installing")
            result = conn.run(install_cmd, warn=True, hide=True)
            if result.exited != 0:
                logger.warning(
                    f"Could not install htr2hpc in conda env for {conn.user}: {result.stderr}"
                )
                _delete_dotfile(conn)
                return False
            _write_dotfile(
                conn, status="installed", gitref=desired_gitref, version=desired_version
            )
            logger.info(
                f"htr2hpc {desired_gitref or desired_version} installed for {conn.user}"
            )
            return True

    # Dotfile exists with status=installed — check if version matches
    if _version_matches(dotfile, desired_gitref, desired_version):
        logger.info(
            f"htr2hpc {desired_gitref or desired_version} already installed for {conn.user}, skipping"
        )
        return True

    # Version mismatch — reinstall
    logger.info(
        f"htr2hpc version mismatch for {conn.user}, reinstalling "
        f"(have gitref={dotfile.get('gitref')} version={dotfile.get('version')}, "
        f"want gitref={desired_gitref} version={desired_version})"
    )
    _write_dotfile(conn, status="installing")
    result = conn.run(install_cmd, warn=True, hide=True)
    if result.exited != 0:
        logger.warning(
            f"Could not install htr2hpc in conda env for {conn.user}: {result.stderr}"
        )
        _delete_dotfile(conn)
        return False
    _write_dotfile(
        conn, status="installed", gitref=desired_gitref, version=desired_version
    )
    logger.info(
        f"htr2hpc {desired_gitref or desired_version} reinstalled for {conn.user}"
    )
    return True
