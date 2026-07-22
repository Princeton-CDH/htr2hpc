"""Tests for htr2hpc.users."""
import datetime
from unittest.mock import Mock, patch

from django.contrib.auth.models import User
from django.utils import timezone

from htr2hpc.users import init_user


def make_user(is_staff=False, is_superuser=False, is_active=True, seconds_old=1):
    """Create a mock user with date_joined set to seconds_old seconds ago."""
    user = Mock(spec=User, is_staff=is_staff, is_superuser=is_superuser, is_active=is_active)
    user.date_joined = timezone.now() - datetime.timedelta(seconds=seconds_old)
    return user


def test_new_user_set_inactive():
    user = make_user(seconds_old=1)
    init_user(user, {})
    assert user.is_active is False


def test_existing_user_not_changed():
    user = make_user(seconds_old=60)
    original_active = user.is_active
    init_user(user, {})
    assert user.is_active == original_active


def test_init_user_ignores_user_info():
    user = make_user(seconds_old=1)
    init_user(user, {"uid": "netid123", "mail": "netid@example.com"})
    assert user.is_active is False


def test_init_user_staff_set_inactive():
    # staff/superuser accounts are set inactive by init_user;
    # createcasuser is responsible for activating them after this hook runs
    user = make_user(is_staff=True, seconds_old=1)
    init_user(user, {})
    assert user.is_active is False


def test_init_user_superuser_set_inactive():
    user = make_user(is_superuser=True, seconds_old=1)
    init_user(user, {})
    assert user.is_active is False
