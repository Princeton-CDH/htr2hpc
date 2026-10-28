"""Tests for UsageReportView access and security."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from django.urls import reverse

# mock escriptorium modules before importing htr2hpc views
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())


@pytest.mark.django_db
def test_usage_report_accessible_to_staff(admin_client):
    """Staff can access the usage report."""
    with patch("htr2hpc.views.get_usage_data", return_value=[]):
        response = admin_client.get(reverse("usage-report"))
    assert response.status_code == 200


@pytest.mark.django_db
def test_usage_report_redirects_anonymous(client):
    """Anonymous users are redirected."""
    response = client.get(reverse("usage-report"))
    assert response.status_code == 302
