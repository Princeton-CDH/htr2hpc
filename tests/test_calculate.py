"""Tests for htr2hpc.train.calculate — pure functions, no Django needed."""
import datetime

import pytest

from htr2hpc.train.calculate import (
    calc_cpu_mem,
    calc_full_duration,
    estimate_cpu_mem,
    estimate_duration,
    slurm_count_epoch,
    slurm_get_avg_epoch,
    slurm_get_max_acc,
    stats_get_max_cpu,
)


# ---------------------------------------------------------------------------
# slurm_get_max_acc
# ---------------------------------------------------------------------------

SEGMENT_OUTPUT = (
    "stage 0 foo bar\nval_mean_iu: \n  0.45\n"
    "stage 1 foo bar\nval_mean_iu: \n  0.72\n"
    "stage 2 foo bar\nval_mean_iu: \n  0.61\n"
)

TRANSCRIPTION_OUTPUT = """\
stage 0 foo bar
baz 0.823 0/10
stage 1 foo bar
baz 0.951 0/10
stage 2 foo bar
baz 0.910 0/10
"""


def test_slurm_get_max_acc_segment():
    result = slurm_get_max_acc(SEGMENT_OUTPUT, "Segment")
    assert result == (1, 0.72)


def test_slurm_get_max_acc_transcription():
    result = slurm_get_max_acc(TRANSCRIPTION_OUTPUT, "Recognize")
    assert result == (1, 0.951)


def test_slurm_get_max_acc_empty_returns_none():
    assert slurm_get_max_acc("no matching output here", "Segment") is None
    assert slurm_get_max_acc("no matching output here", "Recognize") is None


# ---------------------------------------------------------------------------
# slurm_count_epoch
# ---------------------------------------------------------------------------

EPOCH_OUTPUT = """\
0:00:05 • epoch 1
0:00:10 • epoch 2
0:00:08 • epoch 3
"""


def test_slurm_count_epoch_multiple():
    assert slurm_count_epoch(EPOCH_OUTPUT) == 3


def test_slurm_count_epoch_single():
    assert slurm_count_epoch("0:01:00 • epoch 1\n") == 1


def test_slurm_count_epoch_none():
    assert slurm_count_epoch("no epoch timestamps here") is None


# ---------------------------------------------------------------------------
# slurm_get_avg_epoch
# ---------------------------------------------------------------------------


def test_slurm_get_avg_epoch_known_durations():
    # Three epochs: 60s, 120s, 90s → avg = ceil(270/3) = 90
    output = "0:01:00 • epoch 1\n0:02:00 • epoch 2\n0:01:30 • epoch 3\n"
    assert slurm_get_avg_epoch(output) == 90


def test_slurm_get_avg_epoch_single():
    output = "0:02:30 • epoch 1\n"
    assert slurm_get_avg_epoch(output) == 150


def test_slurm_get_avg_epoch_none():
    assert slurm_get_avg_epoch("no epoch timestamps") is None


def test_slurm_get_avg_epoch_minimum_one():
    # If all epochs are 0:00:00, result should be 1 (not 0)
    output = "0:00:00 • epoch 1\n0:00:00 • epoch 2\n"
    assert slurm_get_avg_epoch(output) == 1


# ---------------------------------------------------------------------------
# stats_get_max_cpu
# ---------------------------------------------------------------------------


def test_stats_get_max_cpu_mb():
    stats = "some text (1500MB/2000MB per core) more text"
    result = stats_get_max_cpu(stats)
    assert result == pytest.approx(1.5)


def test_stats_get_max_cpu_gb():
    stats = "some text (2.5GB/4.0GB per core) more text"
    result = stats_get_max_cpu(stats)
    assert result == pytest.approx(2.5)


def test_stats_get_max_cpu_no_match():
    assert stats_get_max_cpu("no memory info here") is None


# ---------------------------------------------------------------------------
# calc_full_duration
# ---------------------------------------------------------------------------

SLURM_WITH_EPOCHS = "0:01:00 • epoch 1\n0:01:00 • epoch 2\n0:01:00 • epoch 3\n"
SLURM_NO_EPOCHS = "no epoch timestamps"

JOB_STATS_NORMAL = "Run Time: 0:05:00\n(1000MB/2000MB per core)"
JOB_STATS_LONG = "Run Time: 0:20:00\n(1000MB/2000MB per core)"
JOB_STATS_SHORT = "Run Time: 0:05:00\n(1000MB/2000MB per core)"
JOB_STATS_NO_RUNTIME = "(1000MB/2000MB per core)"


def test_calc_full_duration_normal():
    # 3 epochs completed, avg 60s each, job ran 5 min (300s)
    # setup_time = 300 - (60 * 3) = 120s
    # epoch_request = 50 - 3 = 47
    # epoch_time_est = 47 (>= 11)
    # duration = ceil((120 + 60 * 47 * 1.1) / 60) minutes
    epoch_request, duration = calc_full_duration(SLURM_WITH_EPOCHS, JOB_STATS_NORMAL)
    assert epoch_request == 47
    assert isinstance(duration, datetime.timedelta)
    assert duration.total_seconds() > 0


def test_calc_full_duration_near_50_epochs():
    # Simulate 42 epochs completed → epoch_request = 8 < 11
    # Should return epoch_request=5 and use epoch_time_est=15
    many_epochs = "".join(f"0:01:00 • epoch {i}\n" for i in range(42))
    job_stats = "Run Time: 0:50:00\n"
    epoch_request, duration = calc_full_duration(many_epochs, job_stats)
    assert epoch_request == 5
    assert isinstance(duration, datetime.timedelta)


def test_calc_full_duration_no_epochs_long_job():
    # No epochs completed, job ran > 14 min → assume 15 min/epoch
    epoch_request, duration = calc_full_duration(SLURM_NO_EPOCHS, JOB_STATS_LONG)
    assert epoch_request == 50
    expected = datetime.timedelta(minutes=15 * 51 * 1.1)
    assert duration == expected


def test_calc_full_duration_no_epochs_short_job():
    # No epochs completed, job ran < 14 min → (None, None)
    epoch_request, duration = calc_full_duration(SLURM_NO_EPOCHS, JOB_STATS_SHORT)
    assert epoch_request is None
    assert duration is None


def test_calc_full_duration_no_runtime():
    # No "Run Time:" in job_stats → (None, None)
    epoch_request, duration = calc_full_duration(SLURM_WITH_EPOCHS, JOB_STATS_NO_RUNTIME)
    assert epoch_request is None
    assert duration is None


# ---------------------------------------------------------------------------
# calc_cpu_mem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stats, expected",
    [
        ("(1000MB/2000MB per core)", "2G"),   # 1.0 GB → ceil(1.0 + 0.3) = 2
        ("(1500MB/2000MB per core)", "2G"),   # 1.5 GB → ceil(1.5 + 0.3) = 2
        ("(1800MB/2000MB per core)", "3G"),   # 1.8 GB → ceil(1.8 + 0.3) = 3
        ("(2.5GB/4.0GB per core)", "3G"),     # 2.5 GB → ceil(2.5 + 0.3) = 3
        ("(3.8GB/8.0GB per core)", "5G"),     # 3.8 GB → ceil(3.8 + 0.3) = 5
    ],
)
def test_calc_cpu_mem(stats, expected):
    assert calc_cpu_mem(stats) == expected


def test_calc_cpu_mem_no_match():
    assert calc_cpu_mem("no memory info") is None


# ---------------------------------------------------------------------------
# estimate_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, mode, expected_minutes",
    [
        (10_000_000, "Segment", 5),
        (25_000_000, "Segment", 15),
        (30_000_000, "Recognize", 5),
        (60_000_000, "Recognize", 15),
    ],
)
def test_estimate_duration(size, mode, expected_minutes):
    result = estimate_duration(size, mode)
    assert result == datetime.timedelta(minutes=expected_minutes)


# ---------------------------------------------------------------------------
# estimate_cpu_mem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, mode, expected",
    [
        # Segment thresholds
        (5_000_000, "Segment", "1G"),
        (15_000_000, "Segment", "2G"),
        (30_000_000, "Segment", "3G"),
        (80_000_000, "Segment", "4G"),
        (150_000_000, "Segment", "5G"),
        (200_000_000, "Segment", "6G"),
        (300_000_000, "Segment", "7G"),
        # Recognize thresholds
        (30_000_000, "Recognize", "1G"),
        (60_000_000, "Recognize", "2G"),
    ],
)
def test_estimate_cpu_mem(size, mode, expected):
    assert estimate_cpu_mem(size, mode) == expected
