"""
Pytest fixtures shared across all htr2hpc test modules.

Path setup and module mocks are handled in the root conftest.py via
pytest_configure, which fires before pytest-django calls django.setup().
"""

import pytest
from django.contrib.auth import get_user_model


@pytest.fixture
def user(db):
    """A basic active user."""
    User = get_user_model()
    return User.objects.create_user(
        username="testuser",
        email="testuser@example.com",
        password="testpass123",
        is_active=True,
    )


@pytest.fixture
def admin_user(db):
    """A superuser for admin view tests."""
    User = get_user_model()
    return User.objects.create_superuser(
        username="admin",
        email="admin@example.com",
        password="adminpass123",
    )


@pytest.fixture
def api_client_instance():
    """An eScriptoriumAPIClient pointed at a fake base URL."""
    from htr2hpc.api_client import eScriptoriumAPIClient

    return eScriptoriumAPIClient(
        base_url="https://escriptorium.example.com",
        api_token="test-token-abc123",
    )


@pytest.fixture
def mock_ldap(mocker):
    """Patch LDAPSearch so no real LDAP calls are made."""
    mock_search = mocker.patch("htr2hpc.users.LDAPSearch")
    mock_instance = mock_search.return_value
    # Default: find_user succeeds (returns a non-None result)
    mock_instance.find_user.return_value = {"uid": "testnetid"}
    return mock_instance


@pytest.fixture
def mock_user_info_from_ldap(mocker):
    """Patch user_info_from_ldap to avoid real LDAP attribute population."""
    return mocker.patch("htr2hpc.users.user_info_from_ldap")
