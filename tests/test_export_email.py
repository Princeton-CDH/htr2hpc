"""Tests for the export notification email retention line."""

import pytest
from django.template.loader import render_to_string
from django.test import override_settings

from htr2hpc.templatetags.htr2hpc_tags import absolute_export_url

CONTEXT = {
    "domain": "example.com",
    "export_uri": "users/1/export_doc1_alto_20260901.zip",
}


@pytest.mark.parametrize(
    "domain,expected_scheme",
    [
        ("example.com", "https://"),
        ("https://example.com", "https://"),
    ],
)
@override_settings(DEBUG=False, MEDIA_URL="/media/")
def test_absolute_export_url_uses_https(domain, expected_scheme):
    url = absolute_export_url(domain, "users/1/test.zip")
    assert url.startswith(expected_scheme)
    assert "users/1/test.zip" in url
    assert "/media/" in url


@override_settings(DEBUG=True, MEDIA_URL="/media/")
def test_absolute_export_url_uses_http_in_debug():
    url = absolute_export_url("example.com", "users/1/test.zip")
    assert url.startswith("http://")


@pytest.mark.parametrize(
    "template", ["export/email/ready_message.txt", "export/email/ready_html.html"]
)
def test_retention_line_included(template):
    body = render_to_string(template, CONTEXT)
    assert "This download will expire" in body
    assert "from now" in body


@pytest.mark.parametrize(
    "template", ["export/email/ready_message.txt", "export/email/ready_html.html"]
)
@override_settings(EXPORT_FILE_RETENTION=0)
def test_retention_line_omitted_when_disabled(template):
    body = render_to_string(template, CONTEXT)
    assert "expire" not in body


@pytest.mark.parametrize(
    "template", ["export/email/ready_message.txt", "export/email/ready_html.html"]
)
@override_settings(DEBUG=False)
def test_download_link_uses_https(template):
    body = render_to_string(template, CONTEXT)
    assert "https://example.com" in body
    assert CONTEXT["export_uri"] in body
