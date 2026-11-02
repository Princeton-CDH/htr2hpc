"""Tests for htr2hpc.content models and context processor."""

import pytest
from django.db import IntegrityError
from django.test import RequestFactory

from htr2hpc.content.context_processors import page_snippets
from htr2hpc.content.models import ContentSnippet


@pytest.fixture
def rf():
    return RequestFactory()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_snippet_create():
    obj = ContentSnippet.objects.create(
        url_path="/contact/",
        slot="intro",
        body="<p>Hello</p>",
        is_active=True,
    )
    assert obj.pk is not None
    assert str(obj) == "Page intro — /contact/"


@pytest.mark.django_db
def test_snippet_str_global():
    obj = ContentSnippet.objects.create(
        url_path="", slot="before_body", body="<p>Hi</p>"
    )
    assert str(obj) == "Before body (site-wide) — site-wide"


@pytest.mark.django_db
def test_snippet_unique_together():
    ContentSnippet.objects.create(url_path="/contact/", slot="intro", body="first")
    with pytest.raises(IntegrityError):
        ContentSnippet.objects.create(url_path="/contact/", slot="intro", body="second")


@pytest.mark.django_db
def test_snippet_global_blank_url():
    obj = ContentSnippet.objects.create(url_path="", slot="before_body", body="")
    assert obj.pk is not None
    assert obj.url_path == ""


@pytest.mark.django_db
def test_snippet_global_and_page_specific_coexist():
    """A site-wide and page-specific snippet for the same slot are both valid."""
    ContentSnippet.objects.create(url_path="", slot="intro", body="global")
    ContentSnippet.objects.create(url_path="/contact/", slot="intro", body="page")
    assert ContentSnippet.objects.filter(slot="intro").count() == 2


# ---------------------------------------------------------------------------
# Context processor tests
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_page_snippets_returns_active_snippet(rf):
    ContentSnippet.objects.create(
        url_path="/contact/", slot="intro", body="<p>Hi</p>", is_active=True
    )
    request = rf.get("/contact/")
    ctx = page_snippets(request)
    assert "intro" in ctx["page_snippets"]
    assert len(ctx["page_snippets"]["intro"]) == 1


@pytest.mark.django_db
def test_page_snippets_inactive_excluded(rf):
    ContentSnippet.objects.create(
        url_path="/contact/", slot="intro", body="<p>Hi</p>", is_active=False
    )
    request = rf.get("/contact/")
    ctx = page_snippets(request)
    assert "intro" not in ctx["page_snippets"]


@pytest.mark.django_db
def test_page_snippets_other_url_excluded(rf):
    ContentSnippet.objects.create(
        url_path="/models/", slot="intro", body="<p>Hi</p>", is_active=True
    )
    request = rf.get("/contact/")
    ctx = page_snippets(request)
    assert "intro" not in ctx["page_snippets"]


@pytest.mark.django_db
def test_page_snippets_global_appears_on_all_pages(rf):
    ContentSnippet.objects.create(
        url_path="", slot="before_body", body="<p>Banner</p>", is_active=True
    )
    for path in ["/", "/contact/", "/models/", "/profile/"]:
        request = rf.get(path)
        ctx = page_snippets(request)
        assert "before_body" in ctx["page_snippets"]


@pytest.mark.django_db
def test_page_snippets_keyed_by_slot(rf):
    ContentSnippet.objects.create(
        url_path="/", slot="intro", body="intro", is_active=True
    )
    ContentSnippet.objects.create(
        url_path="", slot="before_body", body="before", is_active=True
    )
    request = rf.get("/")
    ctx = page_snippets(request)
    assert set(ctx["page_snippets"].keys()) == {"intro", "before_body"}


@pytest.mark.django_db
def test_page_snippets_empty_when_none_active(rf):
    request = rf.get("/contact/")
    ctx = page_snippets(request)
    assert ctx["page_snippets"] == {}


@pytest.mark.django_db
def test_page_snippets_site_wide_first(rf):
    """Site-wide snippets (blank url_path) should appear before page-specific ones."""
    ContentSnippet.objects.create(
        url_path="", slot="intro", body="global", is_active=True
    )
    ContentSnippet.objects.create(
        url_path="/", slot="intro", body="page", is_active=True
    )
    request = rf.get("/")
    ctx = page_snippets(request)
    snippets = ctx["page_snippets"]["intro"]
    assert len(snippets) == 2
    assert snippets[0].url_path == ""
    assert snippets[1].url_path == "/"


@pytest.mark.django_db
def test_page_snippets_returns_list_per_slot(rf):
    """Context processor returns lists, not single objects."""
    ContentSnippet.objects.create(
        url_path="/", slot="intro", body="page", is_active=True
    )
    request = rf.get("/")
    ctx = page_snippets(request)
    assert isinstance(ctx["page_snippets"]["intro"], list)


def test_page_snippets_returns_empty_when_import_fails(rf):
    """Returns empty dict when ContentSnippet cannot be imported (Wagtail not installed)."""
    import builtins
    import unittest.mock

    request = rf.get("/")
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "htr2hpc.content.models":
            raise ImportError
        return real_import(name, *args, **kwargs)

    with unittest.mock.patch("builtins.__import__", side_effect=mock_import):
        ctx = page_snippets(request)

    assert ctx == {"page_snippets": {}}
