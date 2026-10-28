"""htr2hpc-owned URL patterns, independent of eScriptorium.

This module contains only the URLs added by htr2hpc itself, so it can be
imported in tests without pulling in the full eScriptorium stack.

The combined app URL config (urls.py) extends eScriptorium's urlpatterns
and includes these via include().
"""

from django.contrib.admin.views.decorators import staff_member_required
from django.urls import path

from htr2hpc.views import UsageReportView, remote_user_setup

urlpatterns = [
    path("profile/hpc-setup/", remote_user_setup, name="hpc-setup"),
    path(
        "usage-report/",
        staff_member_required(UsageReportView.as_view()),
        name="usage-report",
    ),
]
