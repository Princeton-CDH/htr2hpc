"""Tests for SiteContent model and {% site_content %} template tag."""

import pytest
from django.db import IntegrityError
from django.template import Context, Engine

from htr2hpc.models import SiteContent

# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_site_content_create():
    obj = SiteContent.objects.create(
        key="test-key", label="Test Label", body="<p>Hello</p>"
    )
    assert obj.pk is not None
    assert str(obj) == "Test Label"
    assert obj.key == "test-key"
    assert obj.body == "<p>Hello</p>"
    assert obj.updated_at is not None


@pytest.mark.django_db
def test_site_content_str_falls_back_to_key_when_label_empty():
    obj = SiteContent.objects.create(key="fallback-key", label="", body="")
    assert str(obj) == "fallback-key"


@pytest.mark.django_db
def test_site_content_key_is_unique():
    SiteContent.objects.create(key="duplicate", label="First", body="")
    with pytest.raises(IntegrityError):
        SiteContent.objects.create(key="duplicate", label="Second", body="")


@pytest.mark.django_db
def test_site_content_blank_body_allowed():
    obj = SiteContent.objects.create(key="empty-body", label="No content yet", body="")
    assert obj.pk is not None
    assert obj.body == ""


@pytest.mark.django_db
def test_site_content_ordering():
    SiteContent.objects.create(key="z-last", label="Z", body="")
    SiteContent.objects.create(key="a-first", label="A", body="")
    SiteContent.objects.create(key="m-middle", label="M", body="")
    keys = list(SiteContent.objects.values_list("key", flat=True))
    assert keys == ["a-first", "m-middle", "z-last"]


# ---------------------------------------------------------------------------
# Template tag tests
# ---------------------------------------------------------------------------


def _render(template_string):
    """Render a template string using an engine that loads htr2hpc_tags."""
    engine = Engine(
        libraries={"htr2hpc_tags": "htr2hpc.templatetags.htr2hpc_tags"},
        builtins=["htr2hpc.templatetags.htr2hpc_tags"],
    )
    t = engine.from_string(template_string)
    return t.render(Context({}))


@pytest.mark.django_db
def test_site_content_tag_returns_body():
    SiteContent.objects.create(key="hello", label="Hello", body="<p>World</p>")
    output = _render('{% load htr2hpc_tags %}{% site_content "hello" %}')
    assert "<p>World</p>" in output


@pytest.mark.django_db
def test_site_content_tag_missing_key_returns_empty():
    output = _render('{% load htr2hpc_tags %}{% site_content "does-not-exist" %}')
    assert output.strip() == ""


@pytest.mark.django_db
def test_site_content_tag_blank_body_returns_empty():
    SiteContent.objects.create(key="blank", label="Blank", body="")
    output = _render('{% load htr2hpc_tags %}{% site_content "blank" %}')
    assert output.strip() == ""


@pytest.mark.django_db
def test_site_content_tag_renders_html_unescaped():
    """mark_safe must be applied so HTML tags are rendered, not escaped."""
    SiteContent.objects.create(key="html", label="HTML", body="<strong>bold</strong>")
    output = _render('{% load htr2hpc_tags %}{% site_content "html" %}')
    assert "<strong>bold</strong>" in output
    assert "&lt;strong&gt;" not in output
