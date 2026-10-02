"""Tests for htr2hpc.views."""

import sys
from unittest.mock import MagicMock, patch

# htr2hpc.views imports htr2hpc.tasks which pulls in eScriptorium and celery;
# mock those modules before importing anything from htr2hpc.views
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())

import pytest  # noqa: E402
from django.contrib.auth.models import AnonymousUser, User  # noqa: E402
from django.test import RequestFactory, override_settings  # noqa: E402

from htr2hpc.views import remote_user_setup  # noqa: E402


@pytest.fixture
def rf():
    return RequestFactory()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="testuser", password="pass")


def test_remote_user_setup_queues_task(rf, user):
    """POST from an authenticated user queues the hpc_user_setup Celery task."""
    request = rf.post("/profile/hpc-setup/")
    request.user = user
    with (
        patch("htr2hpc.views.hpc_user_setup") as mock_task,
        patch("htr2hpc.views.reverse", return_value="/profile/api-key/"),
    ):
        remote_user_setup(request)
    mock_task.delay.assert_called_once_with(user_pk=user.pk)


def test_remote_user_setup_returns_303(rf, user):
    """POST from an authenticated user returns a 303 See Other redirect."""
    request = rf.post("/profile/hpc-setup/")
    request.user = user
    with (
        patch("htr2hpc.views.hpc_user_setup"),
        patch("htr2hpc.views.reverse", return_value="/profile/api-key/"),
    ):
        response = remote_user_setup(request)
    assert response.status_code == 303


@override_settings(ROOT_URLCONF="htr2hpc.test_url_conf")
def test_remote_user_setup_requires_login(rf):
    """Anonymous users are redirected to the login page with 302."""
    request = rf.post("/profile/hpc-setup/")
    request.user = AnonymousUser()
    response = remote_user_setup(request)
    assert response.status_code == 302


def test_remote_user_setup_requires_post(rf, user):
    """GET requests are rejected with 405 Method Not Allowed."""
    request = rf.get("/profile/hpc-setup/")
    request.user = user
    response = remote_user_setup(request)
    assert response.status_code == 405
