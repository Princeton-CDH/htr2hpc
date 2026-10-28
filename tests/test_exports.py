"""Tests for the CSV export functions in exports.py."""

import sys
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from django.contrib.auth import get_user_model

# mock escriptorium modules before importing htr2hpc
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())

from htr2hpc.exports import USAGE_REPORT_FIELDS, user_to_csv_row

User = get_user_model()


@pytest.fixture
def annotated_user(db):
    """A real User with annotated attrs that get_usage_data would add."""
    user = User.objects.create_user(username="alice")
    user.disk_usage = 2048
    user.total_runtime = timedelta(hours=2)
    user.this_week_runtime = timedelta(minutes=30)
    user.this_month_runtime = timedelta(hours=1)
    for field in USAGE_REPORT_FIELDS:
        if field in ("username", "disk_usage_bytes"):
            continue
        attr = field.replace("_seconds", "").replace("_bytes", "")
        if not hasattr(user, attr):
            setattr(user, attr, 0)
    user.hpc_segtrain_total = 3
    user.vm_segment_total = 10
    user.total_tasks = 15
    return user


def test_user_to_csv_row_has_all_fields(annotated_user):
    """user_to_csv_row returns a dict with all USAGE_REPORT_FIELDS keys."""
    row = user_to_csv_row(annotated_user)
    assert set(row.keys()) == set(USAGE_REPORT_FIELDS)


def test_user_to_csv_row_null_runtime_becomes_zero(db):
    """None runtime values are converted to 0 seconds."""
    user = User.objects.create_user(username="bob")
    user.disk_usage = 0
    user.total_runtime = None
    user.this_week_runtime = None
    user.this_month_runtime = None
    for field in USAGE_REPORT_FIELDS:
        if field not in ("username", "disk_usage_bytes"):
            attr = field.replace("_seconds", "").replace("_bytes", "")
            if not hasattr(user, attr):
                setattr(user, attr, 0)
    row = user_to_csv_row(user)
    assert row["total_runtime_seconds"] == 0
    assert row["this_week_runtime_seconds"] == 0
    assert row["this_month_runtime_seconds"] == 0


def test_user_to_csv_row_runtime_converted(annotated_user):
    """Timedelta runtime is converted to integer seconds."""
    row = user_to_csv_row(annotated_user)
    assert row["total_runtime_seconds"] == 7200  # 2 hours


def test_user_to_csv_row_username(annotated_user):
    """Username is passed through correctly."""
    row = user_to_csv_row(annotated_user)
    assert row["username"] == "alice"


def test_user_to_csv_row_hpc_counts(annotated_user):
    """HPC task counts are included in the row."""
    row = user_to_csv_row(annotated_user)
    assert row["hpc_segtrain_total"] == 3
    assert row["vm_segment_total"] == 10
