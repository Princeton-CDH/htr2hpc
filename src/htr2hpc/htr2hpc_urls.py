"""htr2hpc-owned URL patterns, independent of eScriptorium.

This module contains only the URLs added by htr2hpc itself, so it can be
imported in tests without pulling in the full eScriptorium stack.

The combined app URL config (urls.py) extends eScriptorium's urlpatterns
and includes these via include().
"""

from django.apps import apps
from django.urls import include, path

from htr2hpc.views import UsageReportView, remote_user_setup

urlpatterns = [
    path("profile/hpc-setup/", remote_user_setup, name="hpc-setup"),
    path("usage-report/", UsageReportView.as_view(), name="usage-report"),
]

# Wagtail is an optional dependency; when it's installed, settings add the
# htr2hpc.content app, and staff edit content snippets in the Wagtail admin.
if apps.is_installed("htr2hpc.content"):
    from wagtail.admin import urls as wagtailadmin_urls

    urlpatterns += [
        path("cms/", include(wagtailadmin_urls)),
    ]
