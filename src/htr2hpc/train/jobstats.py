"""Per-job HPC resource stats via sacct and Jobstats.

Decoding logic adapted from Princeton Research Computing's job_defense_shield:
https://github.com/PrincetonUniversity/job_defense_shield/blob/main/src/job_defense_shield/efficiency.py
(GPL-2.0; this file is an independent implementation for Apache-2.0 compatibility)
"""

import base64
import csv
import gzip
import json
import logging
import os
import re
import subprocess
from io import StringIO
from typing import IO

logger = logging.getLogger(__name__)

# sacct fields we request, in order
_SACCT_FIELDS = (
    "jobid",
    "user",
    "account",
    "partition",
    "state",
    "start",
    "end",
    "elapsedraw",
    "nnodes",
    "ncpus",
    "alloctres",
    "admincomment",
)

JOB_STATS_FIELDS = [
    "job_id",
    "username",
    "account",
    "partition",
    "state",
    "start",
    "end",
    "elapsed_seconds",
    "nnodes",
    "ncpus",
    "ngpus",
    "cpu_hours",
    "gpu_hours",
    "cpu_memory_used_gb",
    "cpu_memory_alloc_gb",
    "gpu_memory_used_gb_max",
    "gpu_memory_used_gb_sum",
    "gpu_memory_alloc_gb_max",
    "gpu_memory_alloc_gb_sum",
    "cpu_efficiency_pct",
    "gpu_utilization_pct",
]


def decode_admincomment(s: str | None) -> dict | None:
    """Decode a Jobstats admincomment value to a dict.

    The format is ``JS1:`` + base64(gzip(JSON)).  Returns None for
    JS1:Short, JS1:None, empty strings, missing values, or malformed data.
    """
    if not s or s in ("JS1:Short", "JS1:None"):
        return None
    if not s.startswith("JS1:"):
        return None
    try:
        return json.loads(gzip.decompress(base64.b64decode(s[4:])))
    except (ValueError, OSError):
        logger.debug("Failed to decode admincomment: %r", s)
        return None


def gpus_from_alloctres(tres: str) -> int:
    """Return the number of allocated GPUs from an alloctres string.

    Returns 0 if no GPU entry is present.
    """
    match = re.search(r"gres/gpu=(\d+)", tres)
    return int(match.group(1)) if match else 0


def _sum_cpu_memory(nodes: dict) -> tuple[float | None, float | None]:
    """Return (used_gb, alloc_gb) summed across all nodes.

    Returns (None, None) if data is unavailable or malformed.
    """
    used_total = 0.0
    alloc_total = 0.0
    try:
        for node_data in nodes.values():
            used_total += node_data["used_memory"]
            alloc_total += node_data["total_memory"]
    except (KeyError, TypeError):
        return None, None
    # Jobstats reports memory in bytes; convert to GB
    return used_total / 1024**3, alloc_total / 1024**3


def _aggregate_gpu_memory(
    nodes: dict,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (used_max, used_sum, alloc_max, alloc_sum) in GB across all GPUs.

    Returns (None, None, None, None) if no GPU data is available.
    """
    used_values: list[float] = []
    alloc_values: list[float] = []
    try:
        for node_data in nodes.values():
            # gpu_utilization is a dict keyed by GPU index; gpu_used_memory
            # and gpu_total_memory are parallel dicts at the same level
            gpu_used = node_data.get("gpu_used_memory", {})
            gpu_alloc = node_data.get("gpu_total_memory", {})
            if not gpu_used and not gpu_alloc:
                continue
            used_values.extend(gpu_used.values())
            alloc_values.extend(gpu_alloc.values())
    except (AttributeError, TypeError):
        return None, None, None, None

    if not used_values or not alloc_values:
        return None, None, None, None

    # Jobstats reports GPU memory in bytes; convert to GB
    used_max = max(used_values) / 1024**3
    used_sum = sum(used_values) / 1024**3
    alloc_max = max(alloc_values) / 1024**3
    alloc_sum = sum(alloc_values) / 1024**3
    return used_max, used_sum, alloc_max, alloc_sum


def _cpu_efficiency(nodes: dict, elapsed: int, ncpus: int) -> float | None:
    """Return CPU efficiency as a percentage (0-100).

    Calculated as total CPU time used / (elapsed * ncpus).
    """
    if elapsed <= 0 or ncpus <= 0:
        return None
    try:
        total_used = sum(n["total_time"] for n in nodes.values())
    except (KeyError, TypeError):
        return None
    denom = elapsed * ncpus
    return round(100.0 * total_used / denom, 2) if denom else None


def _gpu_utilization(nodes: dict) -> float | None:
    """Return mean GPU utilization as a percentage (0-100) across all GPUs."""
    values: list[float] = []
    try:
        for node_data in nodes.values():
            gpu_util = node_data.get("gpu_utilization", {})
            values.extend(gpu_util.values())
    except (AttributeError, TypeError):
        return None
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def parse_job_stats(row: dict) -> dict:
    """Parse one sacct pipe-delimited row dict into a typed JobStats dict.

    All Jobstats-derived fields are None when stats are unavailable.
    GPU-specific fields are None for CPU-only jobs (ngpus == 0).
    """
    elapsed = int(row["elapsedraw"]) if row.get("elapsedraw", "").isdigit() else 0
    ncpus = int(row["ncpus"]) if row.get("ncpus", "").isdigit() else 0
    nnodes = int(row["nnodes"]) if row.get("nnodes", "").isdigit() else 0
    ngpus = gpus_from_alloctres(row.get("alloctres", ""))

    cpu_hours = round(elapsed * ncpus / 3600, 4) if elapsed and ncpus else None
    gpu_hours = round(elapsed * ngpus / 3600, 4) if (elapsed and ngpus) else None

    stats = decode_admincomment(row.get("admincomment"))
    nodes = stats.get("nodes", {}) if stats else {}

    cpu_used_gb, cpu_alloc_gb = _sum_cpu_memory(nodes) if nodes else (None, None)
    cpu_eff = _cpu_efficiency(nodes, elapsed, ncpus) if nodes else None

    if ngpus:
        gpu_used_max, gpu_used_sum, gpu_alloc_max, gpu_alloc_sum = (
            _aggregate_gpu_memory(nodes) if nodes else (None, None, None, None)
        )
        gpu_util = _gpu_utilization(nodes) if nodes else None
    else:
        gpu_used_max = gpu_used_sum = gpu_alloc_max = gpu_alloc_sum = None
        gpu_util = None

    return {
        "job_id": row.get("jobid", ""),
        "username": row.get("user", ""),
        "account": row.get("account", ""),
        "partition": row.get("partition", ""),
        "state": row.get("state", ""),
        "start": row.get("start", ""),
        "end": row.get("end", ""),
        "elapsed_seconds": elapsed,
        "nnodes": nnodes,
        "ncpus": ncpus,
        "ngpus": ngpus,
        "cpu_hours": cpu_hours,
        "gpu_hours": gpu_hours,
        "cpu_memory_used_gb": round(cpu_used_gb, 4)
        if cpu_used_gb is not None
        else None,
        "cpu_memory_alloc_gb": round(cpu_alloc_gb, 4)
        if cpu_alloc_gb is not None
        else None,
        "gpu_memory_used_gb_max": round(gpu_used_max, 4)
        if gpu_used_max is not None
        else None,
        "gpu_memory_used_gb_sum": round(gpu_used_sum, 4)
        if gpu_used_sum is not None
        else None,
        "gpu_memory_alloc_gb_max": round(gpu_alloc_max, 4)
        if gpu_alloc_max is not None
        else None,
        "gpu_memory_alloc_gb_sum": round(gpu_alloc_sum, 4)
        if gpu_alloc_sum is not None
        else None,
        "cpu_efficiency_pct": cpu_eff,
        "gpu_utilization_pct": gpu_util,
    }


def get_sacct_output(
    start: str,
    end: str,
    usernames: list[str] | None = None,
    account: str | None = None,
) -> str:
    """Run sacct and return raw pipe-delimited output as a string.

    Exactly one of ``usernames`` or ``account`` must be provided.
    ``start`` and ``end`` should be YYYY-MM-DD strings.

    This function is intentionally isolated so tests can patch it.
    """
    if not usernames and not account:
        raise ValueError("Provide at least one of usernames or account")

    fields = ",".join(_SACCT_FIELDS)
    cmd = [
        "sacct",
        "-X",  # one row per job (not per step)
        "-P",  # pipe-delimited output
        "-n",  # no header
        "-S",
        start,
        "-E",
        end,
        "-o",
        fields,
    ]
    if account:
        cmd += ["-A", account]
    if usernames:
        cmd += ["-u", ",".join(usernames)]

    env_extra = {"SLURM_TIME_FORMAT": "standard"}
    env = {**os.environ, **env_extra}

    logger.debug("Running sacct: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    return result.stdout


def sacct_to_job_stats(
    start: str,
    end: str,
    usernames: list[str] | None = None,
    account: str | None = None,
) -> list[dict]:
    """Return a list of parsed job-stat dicts for the given date range and scope.

    Jobs with no Jobstats data are included; their stats fields will be None.
    """
    raw = get_sacct_output(start, end, usernames=usernames, account=account)
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != len(_SACCT_FIELDS):
            logger.warning("Unexpected sacct row (got %d fields): %r", len(parts), line)
            continue
        row = dict(zip(_SACCT_FIELDS, parts, strict=True))
        rows.append(parse_job_stats(row))
    return rows


def job_stats_to_csv(rows: list[dict], dest: IO) -> None:
    """Write job stats rows as CSV to dest (any writable file-like object)."""
    writer = csv.DictWriter(dest, fieldnames=JOB_STATS_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def job_stats_to_csv_string(rows: list[dict]) -> str:
    """Return job stats rows formatted as a CSV string."""
    buf = StringIO()
    job_stats_to_csv(rows, buf)
    return buf.getvalue()
