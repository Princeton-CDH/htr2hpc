"""Tests for the UsageReportView."""

import csv
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

# mock escriptorium modules before importing anything that depends on them
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())

User = get_user_model()

_FAKE_USER_ATTRS = {
    "username": "testuser",
    "disk_usage": 1024,
    "total_tasks": 10,
    "last_week_tasks": 3,
    "last_month_tasks": 7,
    "total_runtime": None,
    "last_week_runtime": None,
    "last_month_runtime": None,
    "total_cpu_usage": 5.0,
    "last_week_cpu_usage": 1.0,
    "last_month_cpu_usage": 3.0,
    "total_gpu_usage": 2.0,
    "last_week_gpu_usage": 0.5,
    "last_month_gpu_usage": 1.5,
    "hpc_segtrain_total": 4,
    "hpc_segtrain_last_week": 1,
    "hpc_segtrain_last_month": 3,
    "hpc_train_total": 2,
    "hpc_train_last_week": 0,
    "hpc_train_last_month": 1,
}


def _make_fake_user(**overrides):
    obj = MagicMock()
    for k, v in {**_FAKE_USER_ATTRS, **overrides}.items():
        setattr(obj, k, v)
    return obj


@pytest.fixture
def staff_user(db):
    return User.objects.create_user(username="admin", password="pass", is_staff=True)


@pytest.fixture
def regular_user(db):
    return User.objects.create_user(username="regular", password="pass")


@pytest.fixture
def staff_client(staff_user):
    client = Client()
    client.force_login(staff_user)
    return client


@pytest.fixture
def regular_client(regular_user):
    client = Client()
    client.force_login(regular_user)
    return client


@pytest.mark.django_db
def test_usage_report_page_accessible_to_staff(staff_client):
    """Staff can access the usage report page."""
    response = staff_client.get(reverse("usage-report"))
    assert response.status_code == 200


@pytest.mark.django_db
def test_usage_report_page_redirects_anonymous():
    """Anonymous users are redirected to login."""
    client = Client()
    response = client.get(reverse("usage-report"))
    assert response.status_code == 302


@pytest.mark.django_db
def test_usage_report_csv_content_type(staff_client):
    """POST returns a CSV file with correct content-type and filename."""
    with patch(
        "htr2hpc.views.UsageReportView.get_usage_data",
        return_value=[_make_fake_user()],
    ):
        response = staff_client.post(reverse("usage-report"))
    assert response.status_code == 200
    assert response["Content-Type"] == "text/csv"
    assert "attachment" in response["Content-Disposition"]
    assert ".csv" in response["Content-Disposition"]


@pytest.mark.django_db
def test_usage_report_csv_has_required_columns(staff_client):
    """CSV includes all required columns."""
    with patch(
        "htr2hpc.views.UsageReportView.get_usage_data",
        return_value=[_make_fake_user()],
    ):
        response = staff_client.post(reverse("usage-report"))
    rows = list(csv.DictReader(StringIO(response.content.decode())))
    assert len(rows) == 1
    assert set(rows[0].keys()) == {
        "username",
        "disk_usage_bytes",
        "total_tasks",
        "last_week_tasks",
        "last_month_tasks",
        "total_runtime_seconds",
        "last_week_runtime_seconds",
        "last_month_runtime_seconds",
        "total_cpu_usage",
        "last_week_cpu_usage",
        "last_month_cpu_usage",
        "total_gpu_usage",
        "last_week_gpu_usage",
        "last_month_gpu_usage",
        "hpc_segtrain_total",
        "hpc_segtrain_last_week",
        "hpc_segtrain_last_month",
        "hpc_train_total",
        "hpc_train_last_week",
        "hpc_train_last_month",
    }


@pytest.mark.django_db
def test_usage_report_csv_values(staff_client):
    """CSV row contains correct values from the queryset."""
    fake = _make_fake_user(username="alice", hpc_segtrain_total=5, total_cpu_usage=5.0)
    with patch("htr2hpc.views.UsageReportView.get_usage_data", return_value=[fake]):
        response = staff_client.post(reverse("usage-report"))
    rows = list(csv.DictReader(StringIO(response.content.decode())))
    assert rows[0]["username"] == "alice"
    assert rows[0]["hpc_segtrain_total"] == "5"
    assert rows[0]["total_cpu_usage"] == "5.0"
