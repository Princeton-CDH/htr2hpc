"""Minimal URL config for tests — does not import escriptorium.urls."""

from django.contrib.auth.decorators import login_required
from django.urls import path

from htr2hpc.views import UsageReportView

urlpatterns = [
    path(
        "usage-report/",
        login_required(UsageReportView.as_view()),
        name="usage-report",
    ),
]
