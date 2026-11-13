"""Tests for the wagtail_config template tag override in htr2hpc.content."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

# the override only exists when the optional Wagtail dependency is installed
pytest.importorskip("wagtail")

from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.urls import NoReverseMatch, reverse
from wagtail.admin.templatetags import wagtailadmin_tags as original_tags

from htr2hpc.content.templatetags import wagtailadmin_tags as custom_tags


@pytest.fixture
def context(rf):
    request = rf.get("/cms/")
    request.user = AnonymousUser()
    return {"request": request}


def test_patched_reverse_skips_unsupported_urls():
    for name in custom_tags.UNSUPPORTED_URLS:
        assert custom_tags.patched_reverse(name) is None


def test_patched_reverse_resolves_other_urls():
    assert custom_tags.patched_reverse("admin:login") == reverse("admin:login")


def test_patched_reverse_still_fails_for_unknown_urls():
    with pytest.raises(NoReverseMatch):
        custom_tags.patched_reverse("not-a-real-url-name")


@pytest.mark.django_db
def test_wagtail_config_omits_documents_and_images(context):
    config = custom_tags.custom_wagtail_config(context)
    assert "DOCUMENTS" not in config["ADMIN_API"]
    assert "IMAGES" not in config["ADMIN_API"]


@pytest.mark.django_db
def test_wagtail_config_keeps_other_entries(context):
    config = custom_tags.custom_wagtail_config(context)
    assert "PAGES" in config["ADMIN_API"]
    assert "ADMIN_URLS" in config
    assert "CSRF_TOKEN" in config


@pytest.mark.django_db
def test_wagtail_config_restores_reverse(context):
    reverse_before = original_tags.reverse
    custom_tags.custom_wagtail_config(context)
    assert original_tags.reverse is reverse_before


@pytest.mark.django_db
def test_wagtail_config_without_admin_api_key(context, monkeypatch):
    # some Wagtail versions don't include ADMIN_API in the config
    monkeypatch.setattr(
        original_tags, "wagtail_config", lambda context: {"CSRF_TOKEN": "x"}
    )
    assert custom_tags.custom_wagtail_config(context) == {"CSRF_TOKEN": "x"}


@pytest.mark.django_db
def test_wagtail_admin_loads_for_staff(client):
    user = get_user_model().objects.create_superuser(
        username="cmsadmin", email="cms@example.com", password="pw"
    )
    client.force_login(user)
    response = client.get("/cms/snippets/")
    assert response.status_code == 200


def test_wagtail_config_patch_is_thread_safe(context, monkeypatch):
    # every concurrent call should see the patched reverse while it runs,
    # and the original should be restored once all of them are done
    reverse_before = original_tags.reverse
    seen = []

    def slow_config(context):
        seen.append(original_tags.reverse is custom_tags.patched_reverse)
        time.sleep(0.02)
        seen.append(original_tags.reverse is custom_tags.patched_reverse)
        return {"ADMIN_API": {}}

    monkeypatch.setattr(original_tags, "wagtail_config", slow_config)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: custom_tags.custom_wagtail_config(context), range(16)))

    assert all(seen)
    assert original_tags.reverse is reverse_before
