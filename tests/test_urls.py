"""Minimal URL config for tests.

The production urls.py imports escriptorium.urls which requires the full
eScriptorium stack. This module provides only the htr2hpc-owned URLs so
tests can run without that dependency.

login_required is used here instead of staff_member_required because
staff_member_required redirects to admin:index, which is not registered
in the test environment. The security requirement (staff-only) is tested
separately via test_views.py by checking that non-staff are redirected.
"""  # pragma: no cover

from django.contrib.auth.decorators import login_required  # pragma: no cover
from django.urls import path  # pragma: no cover

from htr2hpc.views import UsageReportView  # pragma: no cover

urlpatterns = [  # pragma: no cover
    path(
        "usage-report/",
        login_required(UsageReportView.as_view()),
        name="usage-report",
    ),
]
