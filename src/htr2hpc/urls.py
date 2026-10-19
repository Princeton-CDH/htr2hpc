from django.contrib.admin.views.decorators import staff_member_required
from django.urls import include, path
from escriptorium.urls import urlpatterns

from htr2hpc.views import UsageReportView, remote_user_setup

urlpatterns += [
    (path("accounts/", include("pucas.cas_urls"))),
    path("profile/hpc-setup/", remote_user_setup, name="hpc-setup"),
    path(
        "usage-report/",
        staff_member_required(UsageReportView.as_view()),
        name="usage-report",
    ),
]
