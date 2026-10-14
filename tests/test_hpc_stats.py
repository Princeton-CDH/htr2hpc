"""Tests for HpcJobStats model, slurm_job_memory, and report_hpc_usage command."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command

from htr2hpc.models import HpcJobStats
from htr2hpc.train.slurm import slurm_job_memory

User = get_user_model()


# ---------------------------------------------------------------------------
# slurm_job_memory tests
# ---------------------------------------------------------------------------


def test_slurm_job_memory_parses_kb():
    """Parses MaxRSS/AveRSS values in KB correctly."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "4096K|2048K\n512K|256K\n"
    with patch("subprocess.run", return_value=mock_result):
        result = slurm_job_memory(12345)
    assert result["max_memory_mb"] == pytest.approx(4.0)
    assert result["avg_memory_mb"] == pytest.approx(2.0)


def test_slurm_job_memory_parses_gb():
    """Parses MaxRSS values in GB correctly."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "2G|1G\n"
    with patch("subprocess.run", return_value=mock_result):
        result = slurm_job_memory(12345)
    assert result["max_memory_mb"] == pytest.approx(2048.0)


def test_slurm_job_memory_returns_none_on_failure():
    """Returns None values when sacct fails."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    with patch("subprocess.run", return_value=mock_result):
        result = slurm_job_memory(12345)
    assert result["max_memory_mb"] is None
    assert result["avg_memory_mb"] is None


def test_slurm_job_memory_handles_empty_output():
    """Returns None values when sacct returns empty output."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "\n"
    with patch("subprocess.run", return_value=mock_result):
        result = slurm_job_memory(12345)
    assert result["max_memory_mb"] is None
    assert result["avg_memory_mb"] is None


# ---------------------------------------------------------------------------
# HpcJobStats model tests
# ---------------------------------------------------------------------------


@pytest.fixture
def user(db):
    return User.objects.create_user(username="testuser", password="pass")


@pytest.mark.django_db
def test_hpc_job_stats_create(user):
    """HpcJobStats can be created and retrieved."""
    stats = HpcJobStats.objects.create(
        task_report_id=42,
        user=user,
        job_type="segtrain",
        slurm_job_id="3395717",
        max_memory_mb=1024.5,
        avg_memory_mb=512.0,
        scratch_storage_mb=2048.0,
    )
    assert stats.pk is not None
    assert str(stats) == "testuser segtrain job 3395717"


@pytest.mark.django_db
def test_hpc_job_stats_str_uses_task_report_id_when_no_slurm_id(user):
    """__str__ falls back to task_report_id when slurm_job_id is blank."""
    stats = HpcJobStats.objects.create(
        task_report_id=99,
        user=user,
        job_type="train",
    )
    assert str(stats) == "testuser train job 99"


@pytest.mark.django_db
def test_hpc_job_stats_nullable_fields(user):
    """Memory and storage fields are optional."""
    stats = HpcJobStats.objects.create(
        task_report_id=1,
        user=user,
        job_type="segtrain",
    )
    assert stats.max_memory_mb is None
    assert stats.avg_memory_mb is None
    assert stats.scratch_storage_mb is None


# ---------------------------------------------------------------------------
# report_hpc_usage management command tests
# ---------------------------------------------------------------------------


@pytest.fixture
def stats_data(db):
    user1 = User.objects.create_user(username="alice", password="pass")
    user2 = User.objects.create_user(username="bob", password="pass")
    HpcJobStats.objects.create(
        task_report_id=1,
        user=user1,
        job_type="segtrain",
        max_memory_mb=1024.0,
        avg_memory_mb=512.0,
        scratch_storage_mb=2048.0,
    )
    HpcJobStats.objects.create(
        task_report_id=2,
        user=user1,
        job_type="segtrain",
        max_memory_mb=2048.0,
        avg_memory_mb=1024.0,
        scratch_storage_mb=3000.0,
    )
    HpcJobStats.objects.create(
        task_report_id=3,
        user=user2,
        job_type="train",
        max_memory_mb=512.0,
        avg_memory_mb=256.0,
        scratch_storage_mb=500.0,
    )
    return user1, user2


@pytest.mark.django_db
def test_report_hpc_usage_text_output(stats_data):
    """Command outputs text table with correct job counts."""
    out = StringIO()
    call_command("report_hpc_usage", stdout=out)
    output = out.getvalue()
    assert "alice" in output
    assert "bob" in output
    assert "segtrain" in output
    assert "train" in output


@pytest.mark.django_db
def test_report_hpc_usage_filter_by_days(stats_data):
    """--days flag filters output to recent jobs."""
    out = StringIO()
    call_command("report_hpc_usage", days=30, stdout=out)
    output = out.getvalue()
    # stats_data was just created so should appear within 30 days
    assert "alice" in output


@pytest.mark.django_db
def test_report_hpc_usage_filter_by_user(stats_data):
    """--user flag filters output to a single user."""
    out = StringIO()
    call_command("report_hpc_usage", user="alice", stdout=out)
    output = out.getvalue()
    assert "alice" in output
    assert "bob" not in output


@pytest.mark.django_db
def test_report_hpc_usage_csv_format(stats_data):
    """--format csv produces parseable CSV."""
    out = StringIO()
    call_command("report_hpc_usage", format="csv", stdout=out)
    output = out.getvalue()
    assert "username" in output
    assert "alice" in output
    assert "," in output


@pytest.mark.django_db
def test_report_hpc_usage_unknown_user():
    """--user with unknown username prints error to stderr."""
    err = StringIO()
    call_command("report_hpc_usage", user="nobody", stderr=err)
    assert "not found" in err.getvalue()


@pytest.mark.django_db
def test_report_hpc_usage_no_data():
    """Output is clear when no stats exist."""
    out = StringIO()
    call_command("report_hpc_usage", stdout=out)
    assert "No HPC job stats found" in out.getvalue()
