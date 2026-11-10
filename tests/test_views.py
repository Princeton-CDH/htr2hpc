"""Tests for htr2hpc views."""

import csv
import sys
from datetime import date, datetime
from io import StringIO
from unittest.mock import MagicMock

import pytest
from django.contrib.auth.models import AnonymousUser
from django.template import Context, Template
from django.utils import timezone

# mock escriptorium modules before importing htr2hpc views
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())

from escr_stub.reporting.models import TaskReport

from htr2hpc.exports import USAGE_REPORT_FIELDS, VM_SEGMENT_METHOD
from htr2hpc.views import UsageReportView, remote_user_setup

TODAY = date(2026, 11, 9)


@pytest.fixture
def today(mocker):
    mocker.patch("htr2hpc.exports.timezone.localdate", return_value=TODAY)
    mocker.patch("htr2hpc.views.timezone.localdate", return_value=TODAY)
    return TODAY


@pytest.fixture
def staff_user(django_user_model):
    return django_user_model.objects.create_user(username="staff", is_staff=True)


def get_report(rf, user, **params):
    request = rf.get("/usage-report/", params)
    request.user = user
    return UsageReportView.as_view()(request)


def read_csv(response):
    content = b"".join(response.streaming_content).decode()
    return list(csv.DictReader(StringIO(content)))


@pytest.mark.django_db
def test_usage_report_csv(rf, today, staff_user):
    TaskReport.objects.create(
        user=staff_user,
        method=VM_SEGMENT_METHOD,
        started_at=datetime(2026, 10, 6, 12, tzinfo=timezone.get_current_timezone()),
    )
    response = get_report(rf, staff_user, report_type="month", start="2026-09-01")

    assert response.status_code == 200
    assert response["Content-Type"] == "text/csv"
    assert (
        response["Content-Disposition"]
        == 'attachment; filename="htr-usage-monthly-2026-09-01-2026-11-09.csv"'
    )
    [row] = read_csv(response)
    assert list(row) == USAGE_REPORT_FIELDS
    assert row["username"] == "staff"
    assert row["report_start"] == "2026-10-01"


@pytest.mark.django_db
def test_usage_report_default_start(rf, today, staff_user):
    response = get_report(rf, staff_user, report_type="total")
    assert response.status_code == 200
    assert (
        "htr-usage-total-2026-09-01-2026-11-09.csv" in response["Content-Disposition"]
    )
    assert read_csv(response) == []


@pytest.mark.django_db
@pytest.mark.parametrize(
    "params",
    [
        {},
        {"report_type": "year"},
        {"report_type": "month", "start": "not-a-date"},
    ],
)
def test_usage_report_invalid_params(rf, staff_user, params):
    response = get_report(rf, staff_user, **params)
    assert response.status_code == 400


@pytest.mark.django_db
def test_usage_report_requires_staff(rf, django_user_model):
    user = django_user_model.objects.create_user(username="notstaff")
    for request_user in (user, AnonymousUser()):
        response = get_report(rf, request_user, report_type="total")
        # staff_member_required redirects to the admin login page
        assert response.status_code == 302
        assert "/admin/login/" in response["Location"]


@pytest.mark.django_db
def test_usage_report_form_tag():
    html = Template("{% load htr2hpc_tags %}{% usage_report_form %}").render(Context())
    assert 'action="/usage-report/"' in html
    assert 'method="get"' in html
    assert 'name="start"' in html
    assert 'value="2026-09-01"' in html
    assert 'name="report_type"' in html


@pytest.mark.django_db
def test_remote_user_setup(rf, mocker, django_user_model):
    setup_task = mocker.patch("htr2hpc.views.hpc_user_setup")
    mocker.patch("htr2hpc.views.reverse", return_value="/profile/apikey/")
    user = django_user_model.objects.create_user(username="alice")
    request = rf.post("/profile/hpc-setup/")
    request.user = user

    response = remote_user_setup(request)

    setup_task.delay.assert_called_once_with(user_pk=user.pk)
    assert response.status_code == 303
    assert response["Location"] == "/profile/apikey/"
