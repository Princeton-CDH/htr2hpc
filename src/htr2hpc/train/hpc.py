"""Utilities for managing the remote HPC conda environment."""
import logging

logger = logging.getLogger(__name__)

# minimum required Kraken major version for training
REQUIRED_KRAKEN_MAJOR = 6


def ensure_kraken_version(conn):
    """Check the Kraken version in the remote htr2hpc conda env and upgrade
    to the required major version if necessary. Returns True if an upgrade
    was performed, False if the version was already sufficient."""
    check_cmd = (
        "module load anaconda3/2024.6 && "
        "conda run -n htr2hpc python -c "
        '"import kraken; print(kraken.__version__)"'
    )
    result = conn.run(check_cmd, warn=True, hide=True)
    if result.exited != 0:
        logger.warning(
            f"Could not check Kraken version in conda env: {result.stderr}"
        )
        return False

    installed_version = result.stdout.strip()
    try:
        major = int(installed_version.split(".")[0])
    except (ValueError, IndexError):
        logger.warning(
            f"Could not parse Kraken version '{installed_version}'; skipping check"
        )
        return False

    if major >= REQUIRED_KRAKEN_MAJOR:
        logger.debug(
            f"Kraken {installed_version} meets minimum major version "
            f"{REQUIRED_KRAKEN_MAJOR}; no upgrade needed"
        )
        return False

    logger.info(
        f"Kraken {installed_version} is below required major version "
        f"{REQUIRED_KRAKEN_MAJOR}; upgrading"
    )
    upgrade_cmd = (
        "module load anaconda3/2024.6 && "
        f'conda run -n htr2hpc pip install -q "kraken~={REQUIRED_KRAKEN_MAJOR}.0"'
    )
    conn.run(upgrade_cmd)
    return True
