"""Tests for htr2hpc views."""

import csv
import importlib
from datetime import date, datetime
from io import StringIO
from unittest.mock import patch

import pytest
from django.contrib.auth.models import AnonymousUser
from django.template import Context, Template
from django.utils import timezone
from reporting.models import TaskReport

from htr2hpc import htr2hpc_urls
from htr2hpc.exports import USAGE_REPORT_FIELDS, VM_SEGMENT_METHOD
from htr2hpc.views import UsageReportView, remote_user_setup

TODAY = date(2026, 11, 9)


@pytest.fixture
def today():
    # exports and views share django.utils.timezone, so one patch covers both
    with patch("django.utils.timezone.localdate", return_value=TODAY):
        yield TODAY


@pytest.fixture
def staff_user(django_user_model):
    return django_user_model.objects.create_user(
        username="staff", email="staff@example.com", is_staff=True
    )


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
def test_usage_report_no_data(rf, today, staff_user):
    response = get_report(rf, staff_user, report_type="total", start="2026-09-01")
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
        {"report_type": "total"},  # start is required
        {"report_type": "year", "start": "2026-09-01"},
        {"report_type": "month", "start": "not-a-date"},
    ],
)
def test_usage_report_invalid_params(rf, staff_user, params):
    response = get_report(rf, staff_user, **params)
    assert response.status_code == 400


@pytest.mark.django_db
def test_usage_report_requires_staff(rf, django_user_model):
    user = django_user_model.objects.create_user(
        username="notstaff", email="notstaff@example.com"
    )
    for request_user in (user, AnonymousUser()):
        response = get_report(rf, request_user, report_type="total")
        # staff_member_required redirects to the admin login page
        assert response.status_code == 302
        assert "/admin/login/" in response["Location"]


def render_form_tag():
    return Template("{% load htr2hpc_tags %}{% usage_report_form %}").render(Context())


@pytest.mark.django_db
def test_usage_report_form_tag():
    html = render_form_tag()
    assert 'action="/usage-report/"' in html
    assert 'method="get"' in html
    assert 'name="start"' in html
    assert 'value="2026-09-01"' in html
    assert 'name="report_type"' in html
    assert "weeks start on Monday" in html


@pytest.mark.django_db
def test_usage_report_form_tag_no_default_start(settings):
    del settings.USAGE_REPORT_DEFAULT_START
    html = render_form_tag()
    assert 'name="start"' in html
    assert "value=" not in html.split('name="start"')[1].split(">")[0]


@pytest.mark.django_db
@patch("htr2hpc.views.reverse", return_value="/profile/apikey/")
@patch("htr2hpc.views.hpc_user_setup")
def test_remote_user_setup(setup_task, mock_reverse, rf, django_user_model):
    user = django_user_model.objects.create_user(
        username="alice", email="alice@example.com"
    )
    request = rf.post("/profile/hpc-setup/")
    request.user = user

    response = remote_user_setup(request)

    setup_task.delay.assert_called_once_with(user_pk=user.pk)
    assert response.status_code == 303
    assert response["Location"] == "/profile/apikey/"


def test_remote_user_setup_requires_login(rf, mocker):
    setup_task = mocker.patch("htr2hpc.views.hpc_user_setup")
    request = rf.post("/profile/hpc-setup/")
    request.user = AnonymousUser()

    response = remote_user_setup(request)

    # anonymous users are sent to log in, and no setup task is queued
    assert response.status_code == 302
    setup_task.delay.assert_not_called()


@pytest.mark.django_db
def test_remote_user_setup_requires_post(rf, mocker, django_user_model):
    setup_task = mocker.patch("htr2hpc.views.hpc_user_setup")
    request = rf.get("/profile/hpc-setup/")
    request.user = django_user_model.objects.create_user(
        username="alice", email="alice@example.com"
    )

    response = remote_user_setup(request)

    # a GET (e.g. a link or prefetch) must not queue a setup task
    assert response.status_code == 405
    setup_task.delay.assert_not_called()


def test_cms_url_only_added_with_content_app():
    """The Wagtail admin URL is only added when the content app is installed,
    so Wagtail stays an optional dependency."""

    def route_paths():
        return [str(pattern.pattern) for pattern in htr2hpc_urls.urlpatterns]

    try:
        with patch("django.apps.apps.is_installed", return_value=False):
            importlib.reload(htr2hpc_urls)
        assert "cms/" not in route_paths()
    finally:
        # restore the URL patterns for the real installed apps
        importlib.reload(htr2hpc_urls)
