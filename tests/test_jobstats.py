"""Tests for htr2hpc.train.jobstats."""

import base64
import csv
import gzip
import json
from io import StringIO
from unittest.mock import patch

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from htr2hpc.train.jobstats import (
    JOB_STATS_FIELDS,
    decode_admincomment,
    gpus_from_alloctres,
    job_stats_to_csv,
    job_stats_to_csv_string,
    parse_job_stats,
    sacct_to_job_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_admincomment(data: dict) -> str:
    """Build a fake Jobstats admincomment string from a dict.

    Mirrors what Adroit writes: 'JS1:' + base64(gzip(json(data))).
    """
    payload = json.dumps(data).encode()
    compressed = gzip.compress(payload)
    return "JS1:" + base64.b64encode(compressed).decode()


SAMPLE_NODES = {
    "node001": {
        "cpus": 8,
        "total_time": 7200,  # seconds of CPU time used
        "used_memory": 4 * 1024**3,  # 4 GB
        "total_memory": 8 * 1024**3,  # 8 GB
        "gpu_utilization": {"0": 85.0, "1": 72.0},
        "gpu_used_memory": {"0": 6 * 1024**3, "1": 5 * 1024**3},
        "gpu_total_memory": {"0": 16 * 1024**3, "1": 16 * 1024**3},
    }
}

SAMPLE_ADMINCOMMENT = make_admincomment({"nodes": SAMPLE_NODES})

ALLOCTRES_GPU = "billing=8,cpu=8,gres/gpu=2,mem=64G,node=1"
ALLOCTRES_CPU_ONLY = "billing=4,cpu=4,mem=32G,node=1"


def _sacct_row(
    jobid="12345",
    user="testuser",
    account="pustaff",
    partition="gpu",
    state="COMPLETED",
    start="2026-01-01T10:00:00",
    end="2026-01-01T12:00:00",
    elapsedraw="7200",
    nnodes="1",
    ncpus="8",
    alloctres=ALLOCTRES_GPU,
    admincomment=SAMPLE_ADMINCOMMENT,
) -> dict:
    return {
        "jobid": jobid,
        "user": user,
        "account": account,
        "partition": partition,
        "state": state,
        "start": start,
        "end": end,
        "elapsedraw": elapsedraw,
        "nnodes": nnodes,
        "ncpus": ncpus,
        "alloctres": alloctres,
        "admincomment": admincomment,
    }


# ---------------------------------------------------------------------------
# decode_admincomment
# ---------------------------------------------------------------------------


def test_decode_admincomment_valid():
    data = {"nodes": {"node1": {"cpus": 4}}}
    s = make_admincomment(data)
    assert decode_admincomment(s) == data


def test_decode_admincomment_none():
    assert decode_admincomment(None) is None


def test_decode_admincomment_empty_string():
    assert decode_admincomment("") is None


def test_decode_admincomment_js1_short():
    assert decode_admincomment("JS1:Short") is None


def test_decode_admincomment_js1_none():
    assert decode_admincomment("JS1:None") is None


def test_decode_admincomment_no_prefix():
    assert decode_admincomment("some random string") is None


def test_decode_admincomment_malformed_base64():
    # Has JS1: prefix but garbage payload
    assert decode_admincomment("JS1:!!!notbase64!!!") is None


# ---------------------------------------------------------------------------
# gpus_from_alloctres
# ---------------------------------------------------------------------------


def test_gpus_from_alloctres_present():
    assert gpus_from_alloctres(ALLOCTRES_GPU) == 2


def test_gpus_from_alloctres_absent():
    assert gpus_from_alloctres(ALLOCTRES_CPU_ONLY) == 0


def test_gpus_from_alloctres_single_gpu():
    assert gpus_from_alloctres("billing=8,cpu=8,gres/gpu=1,mem=64G,node=1") == 1


def test_gpus_from_alloctres_empty():
    assert gpus_from_alloctres("") == 0


# ---------------------------------------------------------------------------
# parse_job_stats — GPU job with full Jobstats data
# ---------------------------------------------------------------------------


def test_parse_job_stats_full():
    row = _sacct_row()
    result = parse_job_stats(row)

    assert result["job_id"] == "12345"
    assert result["username"] == "testuser"
    assert result["account"] == "pustaff"
    assert result["partition"] == "gpu"
    assert result["state"] == "COMPLETED"
    assert result["elapsed_seconds"] == 7200
    assert result["nnodes"] == 1
    assert result["ncpus"] == 8
    assert result["ngpus"] == 2

    # cpu_hours = 7200 * 8 / 3600 = 16.0
    assert result["cpu_hours"] == pytest.approx(16.0)
    # gpu_hours = 7200 * 2 / 3600 = 4.0
    assert result["gpu_hours"] == pytest.approx(4.0)

    # CPU memory: 4 GB used, 8 GB alloc (single node)
    assert result["cpu_memory_used_gb"] == pytest.approx(4.0)
    assert result["cpu_memory_alloc_gb"] == pytest.approx(8.0)

    # GPU memory: used [6GB, 5GB], alloc [16GB, 16GB]
    assert result["gpu_memory_used_gb_max"] == pytest.approx(6.0)
    assert result["gpu_memory_used_gb_sum"] == pytest.approx(11.0)
    assert result["gpu_memory_alloc_gb_max"] == pytest.approx(16.0)
    assert result["gpu_memory_alloc_gb_sum"] == pytest.approx(32.0)

    # CPU efficiency: 7200 / (7200 * 8) = 12.5%
    assert result["cpu_efficiency_pct"] == pytest.approx(12.5)

    # GPU util: mean(85.0, 72.0) = 78.5%
    assert result["gpu_utilization_pct"] == pytest.approx(78.5)


# ---------------------------------------------------------------------------
# parse_job_stats — no Jobstats data
# ---------------------------------------------------------------------------


def test_parse_job_stats_no_admincomment():
    row = _sacct_row(admincomment="JS1:None")
    result = parse_job_stats(row)

    assert result["elapsed_seconds"] == 7200
    assert result["ngpus"] == 2
    assert result["gpu_hours"] == pytest.approx(4.0)

    # All Jobstats-derived fields are None, never 0
    assert result["cpu_memory_used_gb"] is None
    assert result["cpu_memory_alloc_gb"] is None
    assert result["gpu_memory_used_gb_max"] is None
    assert result["gpu_memory_used_gb_sum"] is None
    assert result["cpu_efficiency_pct"] is None
    assert result["gpu_utilization_pct"] is None


def test_parse_job_stats_short_admincomment():
    row = _sacct_row(admincomment="JS1:Short")
    result = parse_job_stats(row)
    assert result["cpu_memory_used_gb"] is None
    assert result["gpu_memory_used_gb_max"] is None


# ---------------------------------------------------------------------------
# parse_job_stats — CPU-only job
# ---------------------------------------------------------------------------


def test_parse_job_stats_cpu_only():
    # CPU-only job: no GPUs, so GPU columns must all be None
    cpu_nodes = {
        "node001": {
            "cpus": 4,
            "total_time": 3600,
            "used_memory": 2 * 1024**3,
            "total_memory": 4 * 1024**3,
        }
    }
    row = _sacct_row(
        alloctres=ALLOCTRES_CPU_ONLY,
        admincomment=make_admincomment({"nodes": cpu_nodes}),
        ncpus="4",
        elapsedraw="3600",
    )
    result = parse_job_stats(row)

    assert result["ngpus"] == 0
    assert result["gpu_hours"] is None  # not 0.0
    assert result["gpu_memory_used_gb_max"] is None
    assert result["gpu_memory_used_gb_sum"] is None
    assert result["gpu_utilization_pct"] is None

    # CPU fields should still be present
    assert result["cpu_hours"] == pytest.approx(4.0)
    assert result["cpu_memory_used_gb"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# parse_job_stats — zero elapsed (very short job)
# ---------------------------------------------------------------------------


def test_parse_job_stats_zero_elapsed():
    row = _sacct_row(elapsedraw="0")
    result = parse_job_stats(row)
    # Row is kept; compute fields are None rather than division-by-zero
    assert result["elapsed_seconds"] == 0
    assert result["cpu_hours"] is None
    assert result["gpu_hours"] is None
    assert result["cpu_efficiency_pct"] is None


# ---------------------------------------------------------------------------
# sacct_to_job_stats (patches get_sacct_output)
# ---------------------------------------------------------------------------

_SACCT_FIELDS_ORDER = (
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


def _make_sacct_line(row: dict) -> str:
    return "|".join(row[f] for f in _SACCT_FIELDS_ORDER)


def test_sacct_to_job_stats_parses_rows():
    row1 = _sacct_row(jobid="100")
    row2 = _sacct_row(jobid="101", admincomment="JS1:None")
    fake_output = "\n".join([_make_sacct_line(row1), _make_sacct_line(row2)]) + "\n"

    with patch(
        "htr2hpc.train.jobstats.get_sacct_output",
        return_value=fake_output,
    ):
        results = sacct_to_job_stats("2026-01-01", "2026-01-31", account="pustaff")

    assert len(results) == 2
    assert results[0]["job_id"] == "100"
    assert results[1]["job_id"] == "101"
    # Row without Jobstats data keeps the row, with None stats
    assert results[1]["cpu_memory_used_gb"] is None


def test_sacct_to_job_stats_skips_malformed_lines():
    bad_line = "only|three|fields"
    good_line = _make_sacct_line(_sacct_row(jobid="200"))
    fake_output = f"{bad_line}\n{good_line}\n"

    with patch(
        "htr2hpc.train.jobstats.get_sacct_output",
        return_value=fake_output,
    ):
        results = sacct_to_job_stats("2026-01-01", "2026-01-31", account="pustaff")

    assert len(results) == 1
    assert results[0]["job_id"] == "200"


def test_sacct_to_job_stats_empty_output():
    with patch("htr2hpc.train.jobstats.get_sacct_output", return_value=""):
        results = sacct_to_job_stats("2026-01-01", "2026-01-31", account="pustaff")
    assert results == []


# ---------------------------------------------------------------------------
# get_sacct_output raises if no scope given
# ---------------------------------------------------------------------------


def test_get_sacct_output_requires_scope():
    from htr2hpc.train.jobstats import get_sacct_output

    with pytest.raises(ValueError, match="at least one"):
        get_sacct_output("2026-01-01", "2026-01-31")


# ---------------------------------------------------------------------------
# job_stats_to_csv
# ---------------------------------------------------------------------------


def test_job_stats_to_csv_headers():
    buf = StringIO()
    job_stats_to_csv([], buf)
    buf.seek(0)
    reader = csv.DictReader(buf)
    assert reader.fieldnames == JOB_STATS_FIELDS


def test_job_stats_to_csv_row_values():
    row = _sacct_row()
    parsed = [parse_job_stats(row)]
    buf = StringIO()
    job_stats_to_csv(parsed, buf)
    buf.seek(0)
    reader = csv.DictReader(buf)
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["job_id"] == "12345"
    assert rows[0]["username"] == "testuser"
    assert float(rows[0]["gpu_hours"]) == pytest.approx(4.0)


def test_job_stats_to_csv_none_fields_are_empty():
    # None values should appear as empty strings in CSV, not "None"
    row = _sacct_row(admincomment="JS1:None")
    parsed = [parse_job_stats(row)]
    s = job_stats_to_csv_string(parsed)
    reader = csv.DictReader(StringIO(s))
    rows = list(reader)
    assert rows[0]["cpu_memory_used_gb"] == ""
    assert rows[0]["gpu_memory_used_gb_max"] == ""


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_management_command_requires_scope():
    with pytest.raises((CommandError, SystemExit)):
        call_command("hpc_job_stats", start="2026-01-01", end="2026-01-31")


@pytest.mark.django_db
def test_management_command_outputs_csv(capsys):
    row = _sacct_row()
    fake_output = _make_sacct_line(row) + "\n"
    with patch(
        "htr2hpc.train.jobstats.get_sacct_output",
        return_value=fake_output,
    ):
        call_command(
            "hpc_job_stats",
            start="2026-01-01",
            end="2026-01-31",
            account="pustaff",
        )

    captured = capsys.readouterr()
    # stdout should be valid CSV with expected headers
    reader = csv.DictReader(StringIO(captured.out))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["job_id"] == "12345"
    assert reader.fieldnames == JOB_STATS_FIELDS


@pytest.mark.django_db
def test_management_command_writes_file(tmp_path):
    row = _sacct_row()
    fake_output = _make_sacct_line(row) + "\n"
    out_file = tmp_path / "jobs.csv"
    with patch(
        "htr2hpc.train.jobstats.get_sacct_output",
        return_value=fake_output,
    ):
        call_command(
            "hpc_job_stats",
            start="2026-01-01",
            end="2026-01-31",
            account="pustaff",
            output=str(out_file),
        )

    assert out_file.exists()
    with out_file.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["username"] == "testuser"


@pytest.mark.django_db
def test_management_command_active_users_only():
    from django.contrib.auth import get_user_model

    User = get_user_model()
    User.objects.create_user(username="alice", password="x", is_active=True)
    User.objects.create_user(username="bob", password="x", is_active=False)

    with patch(
        "htr2hpc.train.jobstats.get_sacct_output",
        return_value="",
    ) as mock_sacct:
        call_command(
            "hpc_job_stats",
            start="2026-01-01",
            end="2026-01-31",
            account="pustaff",
            active_users_only=True,
        )

    # sacct should have been called with only the active user
    _args, kwargs = mock_sacct.call_args
    assert "alice" in kwargs.get("usernames", [])
    assert "bob" not in kwargs.get("usernames", [])
