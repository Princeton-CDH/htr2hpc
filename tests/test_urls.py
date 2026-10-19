"""Minimal URL config for tests — does not import escriptorium.urls."""  # pragma: no cover

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
