"""Tests for htr2hpc.users — CAS user management admin.

htr2hpc.users is added by the feature/cas-user-admin branch. These tests
are written in advance and will be skipped until that module exists.
"""
import pytest

pytest.importorskip("htr2hpc.users", reason="htr2hpc.users not yet merged into develop")

from django.contrib.admin.sites import AdminSite
from django.contrib.auth import get_user_model
from django.contrib.messages.storage.fallback import FallbackStorage
from django.test import RequestFactory

from htr2hpc.users import CasUserInitForm, Htr2HpcUserAdmin, init_new_user


# ---------------------------------------------------------------------------
# init_new_user
# ---------------------------------------------------------------------------


def test_init_new_user_sets_inactive():
    """New CAS accounts must start inactive so admins can vet them."""
    User = get_user_model()
    user = User(username="newuser", is_active=True)
    init_new_user(user, user_info={})
    assert user.is_active is False


# ---------------------------------------------------------------------------
# CasUserInitForm
# ---------------------------------------------------------------------------


def test_cas_user_init_form_splits_space_separated():
    form = CasUserInitForm(data={"netids": "abc123 def456"})
    assert form.is_valid()
    assert form.cleaned_data["netids"] == ["abc123", "def456"]


def test_cas_user_init_form_splits_newline_separated():
    form = CasUserInitForm(data={"netids": "abc123\ndef456\nghi789"})
    assert form.is_valid()
    assert form.cleaned_data["netids"] == ["abc123", "def456", "ghi789"]


def test_cas_user_init_form_splits_mixed_whitespace():
    form = CasUserInitForm(data={"netids": "abc123 def456\nghi789"})
    assert form.is_valid()
    assert form.cleaned_data["netids"] == ["abc123", "def456", "ghi789"]


def test_cas_user_init_form_empty_is_invalid():
    form = CasUserInitForm(data={"netids": ""})
    assert not form.is_valid()
    assert "netids" in form.errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(method, data=None, user=None):
    """Build a minimal request with message storage attached."""
    rf = RequestFactory()
    request = getattr(rf, method)("/", data=data or {})
    request.user = user
    setattr(request, "session", {})
    setattr(request, "_messages", FallbackStorage(request))
    return request


def _get_messages(request):
    return list(request._messages)


# ---------------------------------------------------------------------------
# Htr2HpcUserAdmin.activate
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_activate_sets_is_active(admin_user):
    User = get_user_model()
    inactive = User.objects.create_user(
        username="inactive",
        email="inactive@example.com",
        password="x",
        is_active=False,
    )
    site = AdminSite()
    admin_instance = Htr2HpcUserAdmin(User, site)
    request = _make_request("post", user=admin_user)

    admin_instance.activate(request, User.objects.filter(pk=inactive.pk))

    inactive.refresh_from_db()
    assert inactive.is_active is True


@pytest.mark.django_db
def test_activate_sends_success_message(admin_user):
    User = get_user_model()
    inactive = User.objects.create_user(
        username="inactive2",
        email="inactive2@example.com",
        password="x",
        is_active=False,
    )
    site = AdminSite()
    admin_instance = Htr2HpcUserAdmin(User, site)
    request = _make_request("post", user=admin_user)

    admin_instance.activate(request, User.objects.filter(pk=inactive.pk))

    msgs = _get_messages(request)
    assert len(msgs) == 1
    assert "1 user activated" in str(msgs[0])


# ---------------------------------------------------------------------------
# Htr2HpcUserAdmin.cas_user_init — POST
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_cas_user_init_creates_new_user(admin_user, mock_ldap, mock_user_info_from_ldap):
    User = get_user_model()
    site = AdminSite()
    admin_instance = Htr2HpcUserAdmin(User, site)
    request = _make_request("post", data={"netids": "newnetid"}, user=admin_user)

    admin_instance.cas_user_init(request)

    assert User.objects.filter(username="newnetid").exists()
    mock_ldap.find_user.assert_called_once_with("newnetid")
    mock_user_info_from_ldap.assert_called_once()


@pytest.mark.django_db
def test_cas_user_init_skips_existing_user(admin_user, mock_ldap, mock_user_info_from_ldap):
    """Existing users should not have user_info_from_ldap called (avoids resetting is_active)."""
    User = get_user_model()
    existing = User.objects.create_user(
        username="existingnetid",
        email="existing@example.com",
        password="x",
        is_active=True,
    )
    site = AdminSite()
    admin_instance = Htr2HpcUserAdmin(User, site)
    request = _make_request("post", data={"netids": "existingnetid"}, user=admin_user)

    admin_instance.cas_user_init(request)

    # user_info_from_ldap must NOT be called for existing users
    mock_user_info_from_ldap.assert_not_called()
    # User should still exist and is_active should be unchanged
    existing.refresh_from_db()
    assert existing.is_active is True


@pytest.mark.django_db
def test_cas_user_init_ldap_not_found(admin_user, mock_ldap):
    """NetIDs not found in LDAP must not create a DB record."""
    from pucas.ldap import LDAPSearchException

    mock_ldap.find_user.side_effect = LDAPSearchException("not found")

    User = get_user_model()
    site = AdminSite()
    admin_instance = Htr2HpcUserAdmin(User, site)
    request = _make_request("post", data={"netids": "badnetid"}, user=admin_user)

    admin_instance.cas_user_init(request)

    assert not User.objects.filter(username="badnetid").exists()
    msgs = _get_messages(request)
    assert any("badnetid" in str(m) for m in msgs)


@pytest.mark.django_db
def test_cas_user_init_mixed_valid_and_invalid(admin_user, mock_ldap, mock_user_info_from_ldap):
    """Valid NetIDs are created; invalid ones are reported as errors."""
    from pucas.ldap import LDAPSearchException

    def find_user_side_effect(netid):
        if netid == "badnetid":
            raise LDAPSearchException("not found")
        return {"uid": netid}

    mock_ldap.find_user.side_effect = find_user_side_effect

    User = get_user_model()
    site = AdminSite()
    admin_instance = Htr2HpcUserAdmin(User, site)
    request = _make_request(
        "post", data={"netids": "goodnetid badnetid"}, user=admin_user
    )

    admin_instance.cas_user_init(request)

    assert User.objects.filter(username="goodnetid").exists()
    assert not User.objects.filter(username="badnetid").exists()


# ---------------------------------------------------------------------------
# Htr2HpcUserAdmin.cas_user_init — GET
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_cas_user_init_get_returns_form(admin_user):
    User = get_user_model()
    site = AdminSite()
    admin_instance = Htr2HpcUserAdmin(User, site)
    request = _make_request("get", user=admin_user)

    response = admin_instance.cas_user_init(request)

    assert response.status_code == 200
    assert "form" in response.context_data


# ---------------------------------------------------------------------------
# Htr2HpcUserAdmin.changelist_view
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_changelist_view_adds_cas_init_url(admin_user):
    User = get_user_model()
    site = AdminSite()
    admin_instance = Htr2HpcUserAdmin(User, site)
    request = _make_request("get", user=admin_user)

    response = admin_instance.changelist_view(request)

    assert response.context_data.get("cas_init_url") == "cas-init/"
