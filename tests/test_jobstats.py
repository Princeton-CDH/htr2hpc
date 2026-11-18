"""Tests for per-job HPC stats parsing in htr2hpc.train.jobstats."""

import base64
import gzip
import json
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from htr2hpc.train import jobstats
from htr2hpc.train.jobstats import (
    JOB_STATS_FIELDS,
    decode_admincomment,
    get_job_stats,
    job_stats_to_csv,
    parse_job,
    parse_job_name,
    parse_sacct_json,
    run_sacct,
    sacct_command,
)

GB = 1024**3


def encode_jobstats(data: dict) -> str:
    return "JS1:" + base64.b64encode(gzip.compress(json.dumps(data).encode())).decode()


# decoded Jobstats for a job with one node and one GPU
JOBSTATS = {
    "nodes": {
        "adroit-h11g2": {
            "total_memory": 24 * GB,
            "used_memory": 3128582144,
            "total_time": 1148.6,
            "cpus": 8,
            "gpu_total_memory": {"0": 21072183296},
            "gpu_used_memory": {"0": 5001773056},
        }
    },
    "total_time": 462,
    "gpus": 1,
}


def make_job(**overrides) -> dict:
    """A job in the structure of ``sacct --json`` output (fields we use)."""
    job = {
        "job_id": 3395880,
        "name": "calibrate_segtrain:task2",
        "user": "ht8933",
        "cluster": "adroit",
        "comment": {"administrator": encode_jobstats(JOBSTATS)},
        "state": {"current": ["COMPLETED"]},
        "time": {
            "elapsed": 462,
            "start": 1791388074,
            "limit": {"set": True, "infinite": False, "number": 15},
        },
        "tres": {
            "allocated": [
                {"type": "cpu", "name": "", "count": 8},
                {"type": "mem", "name": "", "count": 24576},
                {"type": "node", "name": "", "count": 1},
                {"type": "billing", "name": "", "count": 14},
                {"type": "gres", "name": "gpu", "count": 1},
            ]
        },
    }
    job.update(overrides)
    return job


# decode_admincomment


def test_decode_admincomment():
    assert decode_admincomment(encode_jobstats(JOBSTATS)) == JOBSTATS


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "JS1:Short",
        "JS1:None",
        "plain text",
        "JS1:not-base64!",
        "JS1:aGVsbG8=",
    ],
)
def test_decode_admincomment_unavailable(value):
    assert decode_admincomment(value) is None


# parse_job_name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("segtrain:mymodel", ("segtrain", "training")),
        ("train:mymodel", ("train", "training")),
        ("calibrate_segtrain:mymodel", ("segtrain", "calibration")),
        ("calibrate_train:mymodel", ("train", "calibration")),
        ("calibrate_train:name:with:colons", ("train", "calibration")),
        ("interactive", None),
        ("my_train:model", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_job_name(name, expected):
    assert parse_job_name(name) == expected


# parse_job


def test_parse_job():
    row = parse_job(make_job())
    assert set(row) == set(JOB_STATS_FIELDS)
    assert row == {
        "username": "ht8933",
        "job_id": 3395880,
        "train_type": "segtrain",
        "phase": "calibration",
        "cluster": "adroit",
        "job_start": "2026-10-07T15:47:54+00:00",
        "time_alloc": 900,  # 15 minutes
        "time_used": 462,
        "exit_code": "COMPLETED",
        "cpu_hours": round(462 * 8 / 3600, 4),
        "gpu_hours": round(462 / 3600, 4),
        "cpu_cores": 8,
        "cpu_mem_alloc_gb": 24.0,
        "cpu_mem_used_gb": round(3128582144 / GB, 4),
        "gpu_mem_alloc_gb": round(21072183296 / GB, 4),
        "gpu_mem_used_gb": round(5001773056 / GB, 4),
    }


def test_parse_job_not_htr2hpc():
    assert parse_job(make_job(name="my-notebook")) is None


def test_parse_job_not_started():
    job = make_job()
    job["time"]["start"] = 0
    assert parse_job(job) is None


@pytest.mark.parametrize("value", ["JS1:Short", "JS1:None", "", "JS1:garbage"])
def test_parse_job_no_jobstats(value):
    row = parse_job(make_job(comment={"administrator": value}))
    # allocation numbers are still reported
    assert row["cpu_hours"] > 0
    assert row["gpu_hours"] > 0
    assert row["cpu_mem_alloc_gb"] == 24.0
    # jobstats fields are blank
    assert row["cpu_mem_used_gb"] is None
    assert row["gpu_mem_alloc_gb"] is None
    assert row["gpu_mem_used_gb"] is None


def test_parse_job_jobstats_without_gpu_data():
    stats = {"nodes": {"n1": {"used_memory": GB, "total_memory": 4 * GB}}}
    row = parse_job(make_job(comment={"administrator": encode_jobstats(stats)}))
    assert row["cpu_mem_used_gb"] == 1.0
    assert row["gpu_mem_alloc_gb"] is None
    assert row["gpu_mem_used_gb"] is None


def test_parse_job_cpu_only():
    job = make_job()
    job["tres"]["allocated"] = [
        t for t in job["tres"]["allocated"] if t["type"] != "gres"
    ]
    row = parse_job(job)
    # blank, not zero
    assert row["gpu_hours"] is None
    assert row["gpu_mem_alloc_gb"] is None
    assert row["gpu_mem_used_gb"] is None
    assert row["cpu_hours"] is not None


@pytest.mark.parametrize("state", ["TIMEOUT", "OUT_OF_MEMORY", "FAILED"])
def test_parse_job_exit_state(state):
    assert parse_job(make_job(state={"current": [state]}))["exit_code"] == state


def test_parse_job_unlimited_time():
    job = make_job()
    job["time"]["limit"] = {"set": False, "infinite": True, "number": 0}
    assert parse_job(job)["time_alloc"] is None


# parse_sacct_json


def test_parse_sacct_json_sorted_and_filtered():
    later = make_job(job_id=2)
    later["time"] = {**later["time"], "start": 1791390000}
    earlier = make_job(job_id=1)
    other = make_job(job_id=3, name="jupyter")
    output = json.dumps({"jobs": [later, other, earlier]})
    rows = parse_sacct_json(output)
    assert [r["job_id"] for r in rows] == [1, 2]


def test_parse_sacct_json_no_jobs():
    assert parse_sacct_json(json.dumps({"jobs": []})) == []
    assert parse_sacct_json("{}") == []


# commands


def test_sacct_command():
    cmd = sacct_command("2026-10-01", "2026-11-01", ["bob", "alice", "bob"])
    assert cmd == [
        "sacct", "--json", "-X", "-S", "2026-10-01", "-E", "2026-11-01", "-u", "alice,bob"
    ]  # fmt: skip
    assert "-a" not in cmd


def test_sacct_command_requires_users():
    with pytest.raises(ValueError, match="username"):
        sacct_command("2026-10-01", "2026-11-01", [])


def test_run_sacct_local():
    with patch.object(jobstats.subprocess, "run") as mock_run:
        mock_run.return_value.stdout = "{}"
        assert run_sacct(["sacct", "--json"]) == "{}"
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == ["sacct", "--json"]


def test_run_sacct_remote():
    conn = Mock()
    conn.run.return_value.stdout = "{}"
    assert run_sacct(["sacct", "-u", "a b"], connection=conn) == "{}"
    # arguments are shell-quoted for the remote shell
    conn.run.assert_called_once_with("sacct -u 'a b'", hide=True)


def test_get_job_stats():
    output = json.dumps({"jobs": [make_job()]})
    with patch.object(jobstats, "run_sacct", return_value=output) as mock_run:
        conn = Mock()
        rows = get_job_stats("2026-10-01", "2026-11-01", ["ht8933"], connection=conn)
    assert len(rows) == 1
    assert mock_run.call_args.args[0] == sacct_command(
        "2026-10-01", "2026-11-01", ["ht8933"]
    )
    assert mock_run.call_args.kwargs == {"connection": conn}


# csv


def test_job_stats_to_csv():
    out = StringIO()
    job_stats_to_csv([parse_job(make_job(comment={"administrator": "JS1:None"}))], out)
    lines = out.getvalue().splitlines()
    assert lines[0] == ",".join(JOB_STATS_FIELDS)
    assert len(lines) == 2
    # blank (not zero or "None") for missing values
    assert lines[1].endswith(",8,24.0,,,")
