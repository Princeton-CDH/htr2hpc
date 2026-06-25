"""Tests for htr2hpc.users."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from htr2hpc.users import CasUserInitForm, init_new_user


# ---------------------------------------------------------------------------
# init_new_user
# ---------------------------------------------------------------------------


def test_init_new_user_sets_inactive():
    user = SimpleNamespace(is_active=True)
    init_new_user(user, {})
    assert user.is_active is False


def test_init_new_user_ignores_user_info():
    user = SimpleNamespace(is_active=True)
    init_new_user(user, {"uid": "netid123", "mail": "netid@example.com"})
    assert user.is_active is False


def test_init_new_user_already_inactive():
    user = SimpleNamespace(is_active=False)
    init_new_user(user, {})
    assert user.is_active is False


# ---------------------------------------------------------------------------
# CasUserInitForm
# ---------------------------------------------------------------------------


def test_casuserinit_form_splits_spaces():
    form = CasUserInitForm({"netids": "abc123 def456"})
    assert form.is_valid()
    assert form.cleaned_data["netids"] == ["abc123", "def456"]


def test_casuserinit_form_splits_newlines():
    form = CasUserInitForm({"netids": "abc123\ndef456\nghi789"})
    assert form.is_valid()
    assert form.cleaned_data["netids"] == ["abc123", "def456", "ghi789"]


def test_casuserinit_form_splits_mixed_whitespace():
    form = CasUserInitForm({"netids": "abc123\ndef456 ghi789"})
    assert form.is_valid()
    assert form.cleaned_data["netids"] == ["abc123", "def456", "ghi789"]


def test_casuserinit_form_single_netid():
    form = CasUserInitForm({"netids": "abc123"})
    assert form.is_valid()
    assert form.cleaned_data["netids"] == ["abc123"]


def test_casuserinit_form_empty_is_invalid():
    form = CasUserInitForm({"netids": ""})
    assert not form.is_valid()
    assert "netids" in form.errors


# ---------------------------------------------------------------------------
# activate admin action
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_activate_sets_users_active():
    from django.contrib.auth import get_user_model
    from htr2hpc.users import Htr2HpcUserAdmin

    User = get_user_model()
    user1 = User.objects.create_user(username="u1", is_active=False)
    user2 = User.objects.create_user(username="u2", is_active=False)

    admin = Htr2HpcUserAdmin(User, None)
    request = MagicMock()
    admin.activate(request, User.objects.filter(username__in=["u1", "u2"]))

    user1.refresh_from_db()
    user2.refresh_from_db()
    assert user1.is_active is True
    assert user2.is_active is True


@pytest.mark.django_db
def test_activate_sends_success_message():
    from django.contrib.auth import get_user_model
    from htr2hpc.users import Htr2HpcUserAdmin

    User = get_user_model()
    User.objects.create_user(username="u3", is_active=False)

    admin_instance = Htr2HpcUserAdmin(User, None)
    request = MagicMock()

    with patch.object(admin_instance, "message_user") as mock_message:
        admin_instance.activate(request, User.objects.filter(username="u3"))

    mock_message.assert_called_once()
    message_text = mock_message.call_args[0][1]
    assert "1 user activated" in message_text


@pytest.mark.django_db
def test_activate_does_not_deactivate_already_active_users():
    from django.contrib.auth import get_user_model
    from htr2hpc.users import Htr2HpcUserAdmin

    User = get_user_model()
    user = User.objects.create_user(username="u4", is_active=True)

    admin = Htr2HpcUserAdmin(User, None)
    request = MagicMock()
    admin.activate(request, User.objects.filter(username="u4"))

    user.refresh_from_db()
    assert user.is_active is True


# ---------------------------------------------------------------------------
# cas_user_init view
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_cas_user_init_creates_new_user():
    from django.contrib.auth import get_user_model
    from htr2hpc.users import Htr2HpcUserAdmin

    User = get_user_model()
    admin_instance = Htr2HpcUserAdmin(User, None)

    request = MagicMock()
    request.method = "POST"
    request.POST = {"netids": "newnetid"}

    with patch("htr2hpc.users.LDAPSearch") as mock_ldap_cls, patch(
        "htr2hpc.users.user_info_from_ldap"
    ) as mock_user_info, patch("htr2hpc.users.redirect"):
        mock_ldap_cls.return_value.find_user.return_value = {"uid": "newnetid"}
        admin_instance.cas_user_init(request)

    assert User.objects.filter(username="newnetid").exists()
    assert mock_user_info.called


@pytest.mark.django_db
def test_cas_user_init_skips_existing_user():
    from django.contrib.auth import get_user_model
    from htr2hpc.users import Htr2HpcUserAdmin

    User = get_user_model()
    existing = User.objects.create_user(username="existingnetid", is_active=True)

    admin_instance = Htr2HpcUserAdmin(User, None)
    request = MagicMock()
    request.method = "POST"
    request.POST = {"netids": "existingnetid"}

    with patch("htr2hpc.users.LDAPSearch") as mock_ldap_cls, patch(
        "htr2hpc.users.user_info_from_ldap"
    ) as mock_user_info, patch("htr2hpc.users.redirect"):
        mock_ldap_cls.return_value.find_user.return_value = {"uid": "existingnetid"}
        admin_instance.cas_user_init(request)

    # user_info_from_ldap should NOT be called for pre-existing users
    assert not mock_user_info.called
    existing.refresh_from_db()
    assert existing.is_active is True


@pytest.mark.django_db
def test_cas_user_init_handles_ldap_error():
    from django.contrib.auth import get_user_model
    from htr2hpc.users import Htr2HpcUserAdmin, LDAPSearchException

    User = get_user_model()
    admin_instance = Htr2HpcUserAdmin(User, None)

    request = MagicMock()
    request.method = "POST"
    request.POST = {"netids": "unknownnetid"}

    with patch("htr2hpc.users.LDAPSearch") as mock_ldap_cls, patch(
        "htr2hpc.users.redirect"
    ):
        mock_ldap_cls.return_value.find_user.side_effect = LDAPSearchException
        admin_instance.cas_user_init(request)

    assert not User.objects.filter(username="unknownnetid").exists()
