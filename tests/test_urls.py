"""URL config for tests — imports htr2hpc_urls directly without escriptorium."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    # admin login is where staff_member_required redirects non-staff users
    path("admin/", admin.site.urls),
    path("", include("htr2hpc.htr2hpc_urls")),
]
