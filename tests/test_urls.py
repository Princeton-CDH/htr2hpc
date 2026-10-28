"""URL config for tests — imports htr2hpc_urls directly without escriptorium."""  # pragma: no cover

from django.contrib import admin  # pragma: no cover
from django.urls import include, path  # pragma: no cover

urlpatterns = [  # pragma: no cover
    path("admin/", admin.site.urls),
    path("", include("htr2hpc.htr2hpc_urls")),
]
