"""Per-job HPC resource stats from ``sacct --json`` and Jobstats.

Slurm reports allocation, timing and state for each job.  Memory actually
used comes from Jobstats, Princeton Research Computing's job statistics tool,
which stores a summary in each job's administrator comment as
``JS1:`` + base64(gzip(JSON)).  The decoding follows the format used by RC's
job_defense_shield project (``efficiency.py``); this is an independent
implementation.

The core functions have no Django dependency, so they can be reused outside
the web app (e.g. by a management command or a scheduled job).
See docs/design-records/242-hpc-usage-stats.md.
"""

import base64
import contextlib
import csv
import gzip
import json
import logging
import re
import shlex
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import IO, Any

logger = logging.getLogger(__name__)

# htr2hpc slurm job names: [calibrate_]segtrain:<model> or [calibrate_]train:<model>
JOB_NAME_PATTERN = re.compile(
    r"^(?P<calibrate>calibrate_)?(?P<train_type>segtrain|train):"
)

JOB_STATS_FIELDS = [
    "username",
    "job_id",
    "train_type",
    "phase",
    "cluster",
    "job_start",
    "time_alloc",
    "time_used",
    "exit_code",
    "cpu_hours",
    "gpu_hours",
    "cpu_cores",
    "cpu_mem_alloc_gb",
    "cpu_mem_used_gb",
    "gpu_mem_alloc_gb",
    "gpu_mem_used_gb",
]

GB = 1024**3
MB_PER_GB = 1024


def sacct_command(start: str, end: str, usernames: Iterable[str]) -> list[str]:
    """Build the ``sacct`` command line.

    ``start`` and ``end`` are YYYY-MM-DD strings.  Results are limited to the
    given users; querying all users is never done.
    """
    usernames = sorted(set(usernames))
    if not usernames:
        raise ValueError("At least one username is required")
    return [
        "sacct",
        "--json",
        "-X",  # one entry per job, not per job step
        "-S",
        start,
        "-E",
        end,
        "-u",
        ",".join(usernames),
    ]


def run_sacct(command: list[str], connection=None) -> str:
    """Run a ``sacct`` command and return its output.

    If a fabric ``connection`` is given, the command runs on the remote host;
    otherwise it runs locally.  Tests replace this function with recorded output.
    """
    logger.debug("Running sacct: %s", shlex.join(command))
    if connection is not None:
        return connection.run(shlex.join(command), hide=True).stdout
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def decode_admincomment(value: str | None) -> dict | None:
    """Decode a Jobstats value (``JS1:`` + base64(gzip(JSON))) to a dict.

    Returns None for ``JS1:Short``, ``JS1:None``, empty or missing values,
    and values that can't be decoded.
    """
    if not value or value in ("JS1:Short", "JS1:None") or not value.startswith("JS1:"):
        return None
    try:
        return json.loads(gzip.decompress(base64.b64decode(value[4:])))
    except (ValueError, OSError):
        logger.debug("Failed to decode Jobstats value: %r", value)
        return None


def _number(value: Any) -> int | float | None:
    """Return the value of a Slurm JSON number, which is either a plain number
    or an object like ``{"set": true, "infinite": false, "number": 15}``."""
    if isinstance(value, dict):
        if not value.get("set") or value.get("infinite"):
            return None
        value = value.get("number")
    return value if isinstance(value, int | float) else None


def parse_job_name(name: str) -> tuple[str, str] | None:
    """Return (train_type, phase) for an htr2hpc job name, or None if the job
    was not started by htr2hpc."""
    match = JOB_NAME_PATTERN.match(name or "")
    if not match:
        return None
    phase = "calibration" if match["calibrate"] else "training"
    return match["train_type"], phase


def _allocated_tres(job: dict) -> dict:
    """Return allocated cpu count, memory in GB and gpu count from a job's tres."""
    cpus = mem_mb = gpus = 0
    for item in job.get("tres", {}).get("allocated", []):
        count = item.get("count") or 0
        if item.get("type") == "cpu":
            cpus = count
        elif item.get("type") == "mem":
            mem_mb = count
        elif item.get("type") == "gres" and item.get("name") == "gpu":
            gpus = count
    return {"cpus": cpus, "mem_gb": mem_mb / MB_PER_GB, "gpus": gpus}


def _jobstats_memory(stats: dict | None) -> dict:
    """Return CPU and GPU memory (GB) from decoded Jobstats, summed over all
    nodes and GPUs; values are None when unavailable."""
    result = {"cpu_used": None, "gpu_alloc": None, "gpu_used": None}
    nodes = (stats or {}).get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        return result
    with contextlib.suppress(KeyError, TypeError):
        result["cpu_used"] = sum(n["used_memory"] for n in nodes.values()) / GB
    try:
        gpu_used = [v for n in nodes.values() for v in n["gpu_used_memory"].values()]
        gpu_alloc = [v for n in nodes.values() for v in n["gpu_total_memory"].values()]
        if gpu_used and gpu_alloc:
            result["gpu_used"] = sum(gpu_used) / GB
            result["gpu_alloc"] = sum(gpu_alloc) / GB
    except (KeyError, TypeError, AttributeError):
        pass
    return result


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def parse_job(job: dict) -> dict | None:
    """Convert one job from ``sacct --json`` to a report row dict.

    Returns None for jobs that were not started by htr2hpc and for jobs that
    have not started yet.  Fields that are not available are None; GPU fields
    are None when the job has no GPU or Jobstats has no GPU data.
    """
    parsed_name = parse_job_name(job.get("name", ""))
    if parsed_name is None:
        return None
    train_type, phase = parsed_name

    time = job.get("time", {})
    start = _number(time.get("start"))
    if not start:  # pending jobs have no start time
        return None
    elapsed = _number(time.get("elapsed")) or 0
    limit_minutes = _number(time.get("limit"))

    tres = _allocated_tres(job)
    memory = _jobstats_memory(
        decode_admincomment(job.get("comment", {}).get("administrator"))
    )
    has_gpu = tres["gpus"] > 0

    state = job.get("state", {}).get("current") or []
    return {
        "username": job.get("user", ""),
        "job_id": job.get("job_id"),
        "train_type": train_type,
        "phase": phase,
        "cluster": job.get("cluster", ""),
        "job_start": datetime.fromtimestamp(start, tz=UTC).isoformat(),
        # slurm time limits are in minutes
        "time_alloc": limit_minutes * 60 if limit_minutes is not None else None,
        "time_used": elapsed,
        "exit_code": state[0] if state else "",
        "cpu_hours": _rounded(elapsed * tres["cpus"] / 3600),
        "gpu_hours": _rounded(elapsed * tres["gpus"] / 3600) if has_gpu else None,
        "cpu_cores": tres["cpus"],
        "cpu_mem_alloc_gb": _rounded(tres["mem_gb"]),
        "cpu_mem_used_gb": _rounded(memory["cpu_used"]),
        "gpu_mem_alloc_gb": _rounded(memory["gpu_alloc"]) if has_gpu else None,
        "gpu_mem_used_gb": _rounded(memory["gpu_used"]) if has_gpu else None,
    }


def parse_sacct_json(output: str) -> list[dict]:
    """Parse ``sacct --json`` output into report rows for htr2hpc jobs,
    ordered by job start."""
    data = json.loads(output)
    rows = [row for job in data.get("jobs", []) if (row := parse_job(job))]
    return sorted(rows, key=lambda row: (row["job_start"], row["job_id"]))


def get_job_stats(
    start: str, end: str, usernames: Iterable[str], connection=None
) -> list[dict]:
    """Return report rows for htr2hpc jobs run by the given users in a date range."""
    command = sacct_command(start, end, usernames)
    return parse_sacct_json(run_sacct(command, connection=connection))


def job_stats_csv_rows(rows: Iterable[dict]) -> Iterable[dict]:
    """Yield the header row and then each report row, as dicts for a
    :class:`csv.DictWriter` with :data:`JOB_STATS_FIELDS`."""
    yield dict(zip(JOB_STATS_FIELDS, JOB_STATS_FIELDS, strict=True))
    yield from rows


def job_stats_to_csv(rows: Iterable[dict], dest: IO) -> None:
    """Write report rows as CSV to a writable file-like object."""
    writer = csv.DictWriter(dest, fieldnames=JOB_STATS_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
