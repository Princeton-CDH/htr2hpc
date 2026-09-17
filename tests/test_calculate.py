"""Tests for htr2hpc.train.calculate — pure functions, no Django needed."""

import datetime
import subprocess
from pathlib import Path

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

# Fixture using real kraken 6.x SLURM .out format for segmentation training
# (captured from Adroit job 3370036, segtrain_doc293_2026-09-17).
# val_mean_iu appears inline on the stage line (all metrics fit within COLUMNS=200).
# Stage 0 has the highest val_mean_iu (0.274).
SEGMENT_OUTPUT = (
    "stage 0/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9/9 0:00:07 • 0:00:00 2.26it/s"
    " train_loss_step: 0.365 val_accuracy: 0.841 val_mean_acc: 0.841 val_mean_iu: 0.274"
    "         early_stopping: 0/10 0.27399\n"
    "                                                                                  val_freq_iu: 0.810 train_loss_epoch: 0.407\n"
    "stage 1/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9/9 0:00:08 • 0:00:00 2.02it/s"
    " train_loss_step: 0.289 val_accuracy: 0.885 val_mean_acc: 0.885 val_mean_iu: 0.259"
    "         early_stopping: 1/10 0.27399\n"
    "                                                                                  val_freq_iu: 0.785 train_loss_epoch: 0.320\n"
    "stage 2/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9/9 0:00:08 • 0:00:00 2.08it/s"
    " train_loss_step: 0.261 val_accuracy: 0.918 val_mean_acc: 0.918 val_mean_iu: 0.245"
    "         early_stopping: 2/10 0.27399\n"
    "                                                                                  val_freq_iu: 0.789 train_loss_epoch: 0.266\n"
)

# Fixture using kraken 6.x non-TTY output format (produced when NO_COLOR=1 is set,
# as in the SLURM job). val_accuracy: label with value on the next line.
# Controlled values ensure epoch 1 has the highest accuracy.
TRANSCRIPTION_OUTPUT = (
    "stage 0/2 ━━━━━━━━━━━━ 274/274 0:00:19 •    14.01it/s train_loss_step:          \n"
    "                               0:00:00                285.433      0/10 0.82300 \n"
    "                                                      val_accuracy:             \n"
    "                                                      0.823                     \n"
    "                                                      val_word_accuracy:        \n"
    "                                                      0.100                     \n"
    "stage 1/2 ━━━━━━━━━━━━ 274/274 0:00:18 •    14.73it/s train_loss_step:          \n"
    "                               0:00:00                266.089      1/10 0.95100 \n"
    "                                                      val_accuracy:             \n"
    "                                                      0.951                     \n"
    "                                                      val_word_accuracy:        \n"
    "                                                      0.200                     \n"
    "stage 2/2 ━━━━━━━━━━━━ 274/274 0:00:18 •    14.73it/s train_loss_step:          \n"
    "                               0:00:00                319.474      2/10 0.91000 \n"
    "                                                      val_accuracy:             \n"
    "                                                      0.910                     \n"
    "                                                      val_word_accuracy:        \n"
    "                                                      0.300                     \n"
)


def test_slurm_get_max_acc_segment():
    result = slurm_get_max_acc(SEGMENT_OUTPUT, "Segment")
    assert result == (0, 0.274)


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
    epoch_request, duration = calc_full_duration(
        SLURM_WITH_EPOCHS, JOB_STATS_NO_RUNTIME
    )
    assert epoch_request is None
    assert duration is None


# ---------------------------------------------------------------------------
# calc_cpu_mem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stats, expected",
    [
        ("(1000MB/2000MB per core)", "2G"),  # 1.0 GB → ceil(1.0 + 0.3) = 2
        ("(1500MB/2000MB per core)", "2G"),  # 1.5 GB → ceil(1.5 + 0.3) = 2
        ("(1800MB/2000MB per core)", "3G"),  # 1.8 GB → ceil(1.8 + 0.3) = 3
        ("(2.5GB/4.0GB per core)", "3G"),  # 2.5 GB → ceil(2.5 + 0.3) = 3
        ("(3.8GB/8.0GB per core)", "5G"),  # 3.8 GB → ceil(3.8 + 0.3) = 5
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


# ---------------------------------------------------------------------------
# integration: slurm_get_max_acc against real ketos output
# ---------------------------------------------------------------------------

RESOURCES = Path(__file__).parent / "resources"


def test_slurm_get_max_acc_recognize_real_output(tmp_path):
    """Run ketos train on a minimal dataset and verify slurm_get_max_acc parses
    real output so that format changes in future kraken releases cause test failures.
    """
    manifest = tmp_path / "train.lst"
    manifest.write_text(str(RESOURCES / "170025120000003,0074-lite.xml") + "\n")

    result = subprocess.run(
        [
            "ketos",
            "-d",
            "cpu",
            "--threads",
            "1",
            "--workers",
            "0",
            "train",
            "-o",
            str(tmp_path / "model"),
            "-f",
            "xml",
            "--spec",
            "[1,12,0,1 Cr3,3,8 S1(1x0)1,3]",
            "--quit",
            "fixed",
            "-N",
            "3",
            "--min-epochs",
            "3",
            "-F",
            "1",
            "-B",
            "1",
            "-t",
            str(manifest),
            "-e",
            str(manifest),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"ketos train failed (exit={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    output = result.stdout + result.stderr
    parsed = slurm_get_max_acc(output, "Recognize")

    assert parsed is not None, (
        f"slurm_get_max_acc returned None; ketos exit={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    epoch, accuracy = parsed
    assert isinstance(epoch, int)
    assert isinstance(accuracy, float)
    assert 0 <= epoch < 3
