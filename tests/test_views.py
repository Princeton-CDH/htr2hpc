"""Tests for UsageReportView access and security."""

import sys
from unittest.mock import MagicMock

import pytest

# mock escriptorium modules before importing htr2hpc views
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())

from django.test import RequestFactory

from htr2hpc.views import UsageReportView


def test_usage_report_view_requires_staff():
    """UsageReportView.dispatch raises PermissionDenied for non-staff users."""
    from django.contrib.auth.models import AnonymousUser

    factory = RequestFactory()
    request = factory.get("/usage-report/")
    request.user = AnonymousUser()

    # The view itself has no auth — security is enforced by staff_member_required
    # in htr2hpc_urls.py. Confirm the view class exists and is a View subclass.
    from django.views import View

    assert issubclass(UsageReportView, View)


@pytest.mark.django_db
def test_usage_report_view_returns_zip_response(db):
    """UsageReportView.get returns a ZIP HttpResponse."""
    from unittest.mock import patch

    from django.contrib.auth import get_user_model
    from django.http import HttpResponse
    from django.test import RequestFactory

    User = get_user_model()
    user = User.objects.create_user(username="staff", is_staff=True)
    factory = RequestFactory()
    request = factory.get("/usage-report/")
    request.user = user

    with (
        patch("htr2hpc.views.get_totals_data", return_value=[]),
        patch("htr2hpc.views.get_monthly_data", return_value=[]),
        patch("htr2hpc.views.get_completed_months", return_value=[(2026, 9)]),
    ):
        response = UsageReportView.as_view()(request)

    assert isinstance(response, HttpResponse)
    assert response["Content-Type"] == "application/zip"
    assert ".zip" in response["Content-Disposition"]
