"""Tests for htr2hpc.content models, context processor and templates."""

import re
from pathlib import Path

import pytest

# content management needs the optional Wagtail dependency; skip these tests
# when it isn't installed (CI runs the tests both with and without Wagtail)
pytest.importorskip("wagtail")

from django.db import IntegrityError
from django.template import Context, Template

from htr2hpc.content.context_processors import page_snippets
from htr2hpc.content.models import ContentSnippet

CONTENT_TEMPLATES = (
    Path(__file__).parent.parent / "src" / "htr2hpc" / "content" / "templates"
)

# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_snippet_defaults_to_live():
    obj = ContentSnippet.objects.create(
        url_path="/contact/", slot="before_main", body="<p>Hello</p>"
    )
    assert obj.live is True


@pytest.mark.django_db
def test_snippet_str():
    page = ContentSnippet.objects.create(
        url_path="/contact/", slot="before_main", body="<p>Hello</p>"
    )
    site_wide = ContentSnippet.objects.create(
        url_path="", slot="after_main", body="<p>Hi</p>"
    )
    assert str(page) == "Before main content (any page) — /contact/"
    assert str(site_wide) == "After main content (any page) — all pages"


@pytest.mark.django_db
def test_snippet_unique_page_and_slot():
    ContentSnippet.objects.create(url_path="/contact/", slot="before_main", body="1")
    with pytest.raises(IntegrityError):
        ContentSnippet.objects.create(
            url_path="/contact/", slot="before_main", body="2"
        )


@pytest.mark.django_db
def test_snippet_unique_applies_to_non_live():
    """The constraint holds whether or not the snippet is live, so no drafts."""
    ContentSnippet.objects.create(
        url_path="/contact/", slot="before_main", body="1", live=False
    )
    with pytest.raises(IntegrityError):
        ContentSnippet.objects.create(
            url_path="/contact/", slot="before_main", body="2"
        )


@pytest.mark.django_db
def test_snippet_site_wide_and_page_specific_coexist():
    ContentSnippet.objects.create(url_path="", slot="before_main", body="site-wide")
    ContentSnippet.objects.create(url_path="/contact/", slot="before_main", body="page")
    assert ContentSnippet.objects.filter(slot="before_main").count() == 2


# ---------------------------------------------------------------------------
# Context processor tests
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_page_snippets_for_page(rf):
    snippet = ContentSnippet.objects.create(
        url_path="/contact/", slot="before_main", body="<p>Hi</p>"
    )
    ctx = page_snippets(rf.get("/contact/"))
    assert ctx["page_snippets"] == {"before_main": [snippet]}


@pytest.mark.django_db
def test_page_snippets_non_live_excluded(rf):
    ContentSnippet.objects.create(
        url_path="/contact/", slot="before_main", body="<p>Hi</p>", live=False
    )
    assert page_snippets(rf.get("/contact/"))["page_snippets"] == {}


@pytest.mark.django_db
def test_page_snippets_other_page_excluded(rf):
    ContentSnippet.objects.create(
        url_path="/models/", slot="before_main", body="<p>Hi</p>"
    )
    assert page_snippets(rf.get("/contact/"))["page_snippets"] == {}


@pytest.mark.django_db
def test_page_snippets_exact_path_match(rf):
    ContentSnippet.objects.create(
        url_path="/contact/", slot="before_main", body="<p>Hi</p>"
    )
    # the query string isn't part of request.path
    assert "before_main" in page_snippets(rf.get("/contact/?x=1"))["page_snippets"]
    # a missing trailing slash doesn't match
    assert page_snippets(rf.get("/contact"))["page_snippets"] == {}


@pytest.mark.django_db
def test_page_snippets_site_wide_on_all_pages(rf):
    ContentSnippet.objects.create(url_path="", slot="before_main", body="<p>Banner</p>")
    for path in ["/", "/contact/", "/models/", "/profile/"]:
        assert "before_main" in page_snippets(rf.get(path))["page_snippets"]


@pytest.mark.django_db
def test_page_snippets_keyed_by_slot_site_wide_first(rf):
    site_wide = ContentSnippet.objects.create(
        url_path="", slot="before_main", body="site-wide"
    )
    page = ContentSnippet.objects.create(url_path="/", slot="before_main", body="page")
    heading = ContentSnippet.objects.create(
        url_path="/", slot="homepage_heading", body="heading"
    )
    ctx = page_snippets(rf.get("/"))
    assert ctx["page_snippets"] == {
        "before_main": [site_wide, page],
        "homepage_heading": [heading],
    }


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


def test_base_template_has_snippet_blocks_around_body():
    """The base.html copy adds pre_body / post_body blocks around the body block,
    inside <div id="main">. (The full template needs eScriptorium to render.)"""
    source = (CONTENT_TEMPLATES / "base.html").read_text()
    positions = [
        source.index('<div id="main"'),
        source.index("{% block pre_body %}"),
        source.index("{% block body %}"),
        source.index("{% block post_body %}"),
        source.index("</section></main>"),
    ]
    assert positions == sorted(positions)
    assert re.search(r"pre_body %}.*page_snippets\.before_main", source)
    assert re.search(r"post_body %}.*page_snippets\.after_main", source)
    assert "MUST be updated whenever eScriptorium is upgraded" in source


@pytest.mark.django_db
def test_homepage_heading(rf):
    ContentSnippet.objects.create(
        url_path="/", slot="homepage_heading", body="<p>Welcome text</p>"
    )
    snippets = page_snippets(rf.get("/"))["page_snippets"]
    html = Template((CONTENT_TEMPLATES / "additional_heading.html").read_text()).render(
        Context({"page_snippets": snippets})
    )
    assert "eScriptorium</h1>" in html
    assert "Welcome text" in html
