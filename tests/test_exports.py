"""Tests for the CSV export functions in exports.py."""

import csv
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from django.urls import reverse

# mock escriptorium modules before importing htr2hpc
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())

from htr2hpc.exports import USAGE_REPORT_FIELDS, user_to_csv_row


def _fake_user(**overrides):
    """Minimal fake annotated user with all required attrs set to 0/None."""
    obj = MagicMock()
    obj.username = "testuser"
    obj.disk_usage = 0
    obj.total_runtime = None
    obj.this_week_runtime = None
    obj.this_month_runtime = None
    for field in USAGE_REPORT_FIELDS:
        if field not in ("username", "disk_usage_bytes"):
            attr = field.replace("_seconds", "").replace("_bytes", "")
            if not hasattr(obj, attr) or isinstance(getattr(obj, attr), MagicMock):
                setattr(obj, attr, 0)
    for k, v in overrides.items():
        setattr(obj, k, v)
    return obj


def test_user_to_csv_row_has_all_fields():
    """user_to_csv_row returns a dict with all USAGE_REPORT_FIELDS keys."""
    row = user_to_csv_row(_fake_user())
    assert set(row.keys()) == set(USAGE_REPORT_FIELDS)


def test_user_to_csv_row_null_runtime_becomes_zero():
    """None runtime values are converted to 0 seconds."""
    row = user_to_csv_row(_fake_user())
    assert row["total_runtime_seconds"] == 0
    assert row["this_week_runtime_seconds"] == 0
    assert row["this_month_runtime_seconds"] == 0


def test_user_to_csv_row_runtime_converted():
    """Timedelta runtime is converted to integer seconds."""
    from datetime import timedelta

    user = _fake_user()
    user.total_runtime = timedelta(hours=1, minutes=30)
    row = user_to_csv_row(user)
    assert row["total_runtime_seconds"] == 5400


def test_user_to_csv_row_username():
    """Username is passed through."""
    row = user_to_csv_row(_fake_user(username="croughan"))
    assert row["username"] == "croughan"


@pytest.mark.django_db
def test_csv_export_columns(admin_client):
    """CSV response contains exactly USAGE_REPORT_FIELDS as column headers."""
    with patch("htr2hpc.views.get_usage_data", return_value=[_fake_user()]):
        response = admin_client.get(reverse("usage-report"))
    content = b"".join(response.streaming_content).decode()
    rows = list(csv.DictReader(StringIO(content)))
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(USAGE_REPORT_FIELDS)
