"""Tests for htr2hpc.users."""
from unittest.mock import Mock

from django.contrib.auth.models import User

from htr2hpc.users import init_user


def test_init_user_sets_inactive():
    user = Mock(spec=User, is_staff=False, is_superuser=False, is_active=True)
    init_user(user, {})
    assert user.is_active is False


def test_init_user_ignores_user_info():
    user = Mock(spec=User, is_staff=False, is_superuser=False, is_active=True)
    init_user(user, {"uid": "netid123", "mail": "netid@example.com"})
    assert user.is_active is False


def test_init_user_already_inactive():
    user = Mock(spec=User, is_staff=False, is_superuser=False, is_active=False)
    init_user(user, {})
    assert user.is_active is False


def test_init_user_staff_stays_active():
    user = Mock(spec=User, is_staff=True, is_superuser=False, is_active=True)
    init_user(user, {})
    assert user.is_active is True


def test_init_user_superuser_stays_active():
    user = Mock(spec=User, is_staff=False, is_superuser=True, is_active=True)
    init_user(user, {})
    assert user.is_active is True
