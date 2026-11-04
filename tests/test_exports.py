"""Tests for the CSV export functions in exports.py."""

import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from django.contrib.auth import get_user_model

# mock escriptorium modules before importing htr2hpc
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())

from htr2hpc.exports import (
    MONTHLY_FIELDS,
    TOTALS_FIELDS,
    get_completed_months,
    monthly_row,
    totals_row,
)

User = get_user_model()


@pytest.fixture
def totals_user(db):
    """A real User with annotated attrs that get_totals_data would add."""
    user = User.objects.create_user(username="alice")
    user.disk_usage = 2048
    user.total_runtime = timedelta(hours=2)
    for field in TOTALS_FIELDS:
        if field in ("username", "disk_usage_bytes"):
            continue
        attr = field.replace("_seconds", "").replace("_bytes", "")
        if not hasattr(user, attr):
            setattr(user, attr, 0)
    user.hpc_segtrain_total = 3
    user.vm_segment_total = 10
    user.total_tasks = 15
    return user


@pytest.fixture
def monthly_user(db):
    """A real User with annotated attrs that get_monthly_data would add."""
    user = User.objects.create_user(username="bob")
    user.runtime = timedelta(minutes=45)
    for field in MONTHLY_FIELDS:
        if field == "username":
            continue
        attr = field.replace("_seconds", "")
        if not hasattr(user, attr):
            setattr(user, attr, 0)
    user.hpc_train = 2
    user.vm_transcribe = 5
    return user


# ---------------------------------------------------------------------------
# totals_row tests
# ---------------------------------------------------------------------------


def test_totals_row_has_all_fields(totals_user):
    """totals_row returns a dict with all TOTALS_FIELDS keys."""
    row = totals_row(totals_user)
    assert set(row.keys()) == set(TOTALS_FIELDS)


def test_totals_row_null_runtime_becomes_zero(db):
    """None runtime values are converted to 0 seconds."""
    user = User.objects.create_user(username="zero")
    user.disk_usage = 0
    user.total_runtime = None
    for field in TOTALS_FIELDS:
        if field not in ("username", "disk_usage_bytes"):
            attr = field.replace("_seconds", "").replace("_bytes", "")
            if not hasattr(user, attr):
                setattr(user, attr, 0)
    row = totals_row(user)
    assert row["total_runtime_seconds"] == 0


def test_totals_row_runtime_converted(totals_user):
    """Timedelta runtime is converted to integer seconds."""
    row = totals_row(totals_user)
    assert row["total_runtime_seconds"] == 7200  # 2 hours


def test_totals_row_username(totals_user):
    """Username is passed through correctly."""
    row = totals_row(totals_user)
    assert row["username"] == "alice"


def test_totals_row_hpc_counts(totals_user):
    """HPC task counts are included in the row."""
    row = totals_row(totals_user)
    assert row["hpc_segtrain_total"] == 3
    assert row["vm_segment_total"] == 10


def test_totals_row_no_gpu_fields(totals_user):
    """GPU usage fields are not present in totals row."""
    row = totals_row(totals_user)
    assert "total_gpu_usage" not in row
    assert "this_week_gpu_usage" not in row


def test_totals_row_no_vm_train_fields(totals_user):
    """vm_train and vm_segtrain are not present in totals row."""
    row = totals_row(totals_user)
    assert "vm_train_total" not in row
    assert "vm_segtrain_total" not in row


# ---------------------------------------------------------------------------
# monthly_row tests
# ---------------------------------------------------------------------------


def test_monthly_row_has_all_fields(monthly_user):
    """monthly_row returns a dict with all MONTHLY_FIELDS keys."""
    row = monthly_row(monthly_user)
    assert set(row.keys()) == set(MONTHLY_FIELDS)


def test_monthly_row_null_runtime_becomes_zero(db):
    """None runtime values are converted to 0 seconds."""
    user = User.objects.create_user(username="zero2")
    user.runtime = None
    for field in MONTHLY_FIELDS:
        if field not in ("username",):
            attr = field.replace("_seconds", "")
            if not hasattr(user, attr):
                setattr(user, attr, 0)
    row = monthly_row(user)
    assert row["runtime_seconds"] == 0


def test_monthly_row_runtime_converted(monthly_user):
    """Timedelta runtime is converted to integer seconds."""
    row = monthly_row(monthly_user)
    assert row["runtime_seconds"] == 2700  # 45 minutes


def test_monthly_row_hpc_counts(monthly_user):
    """HPC task counts are included in monthly row."""
    row = monthly_row(monthly_user)
    assert row["hpc_train"] == 2
    assert row["vm_transcribe"] == 5


# ---------------------------------------------------------------------------
# get_completed_months tests
# ---------------------------------------------------------------------------


def test_get_completed_months_returns_list():
    """Returns a list of (year, month) tuples."""
    months = get_completed_months()
    assert isinstance(months, list)
    for item in months:
        assert len(item) == 2
        _year, month = item
        assert 1 <= month <= 12


def test_get_completed_months_starts_at_report_start():
    """First entry is (2026, 9) — the USAGE_REPORT_START month."""
    months = get_completed_months()
    assert len(months) > 0
    assert months[0] == (2026, 9)


def test_get_completed_months_excludes_current_month():
    """Current month is not included (it's not yet complete)."""
    today = datetime.now(tz=UTC).date()
    months = get_completed_months()
    assert (today.year, today.month) not in months


def test_get_completed_months_sequential():
    """Months are in chronological order with no gaps."""
    from unittest.mock import patch

    # Use Feb 2027 so the list includes a Dec→Jan year rollover
    mock_now = datetime(2027, 2, 15, tzinfo=UTC)
    with patch("htr2hpc.exports.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now
        months = get_completed_months()

    for i in range(1, len(months)):
        prev_year, prev_month = months[i - 1]
        curr_year, curr_month = months[i]
        if prev_month == 12:
            assert curr_year == prev_year + 1 and curr_month == 1
        else:
            assert curr_year == prev_year and curr_month == prev_month + 1


def test_get_completed_months_year_rollover():
    """Months correctly roll over from December to January of the next year."""
    from unittest.mock import patch

    # Mock today to be in Feb 2027 so we get Dec 2026 and Jan 2027 in the list
    mock_now = datetime(2027, 2, 15, tzinfo=UTC)
    with patch("htr2hpc.exports.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now
        months = get_completed_months()

    assert (2026, 12) in months
    assert (2027, 1) in months
    # Dec 2026 should come before Jan 2027
    idx_dec = months.index((2026, 12))
    idx_jan = months.index((2027, 1))
    assert idx_dec < idx_jan


# ---------------------------------------------------------------------------
# Backwards compatibility aliases
# ---------------------------------------------------------------------------


def test_usage_report_fields_alias():
    """USAGE_REPORT_FIELDS is still importable and equals TOTALS_FIELDS."""
    from htr2hpc.exports import USAGE_REPORT_FIELDS

    assert USAGE_REPORT_FIELDS is TOTALS_FIELDS


def test_user_to_csv_row_alias(totals_user):
    """user_to_csv_row is an alias for totals_row."""
    from htr2hpc.exports import user_to_csv_row

    assert user_to_csv_row(totals_user) == totals_row(totals_user)
