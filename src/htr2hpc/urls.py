from django.urls import include, path
from escriptorium.urls import urlpatterns

from htr2hpc.views import HpcJobStatsView, remote_user_setup

urlpatterns += [
    (path("accounts/", include("pucas.cas_urls"))),
    path("profile/hpc-setup/", remote_user_setup, name="hpc-setup"),
    path("usage/hpc-jobs/", HpcJobStatsView.as_view(), name="hpc-job-stats"),
]
