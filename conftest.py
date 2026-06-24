"""
Root conftest.py — shared pytest fixtures for htr2hpc tests.

Path setup and module mocking are handled by the htr2hpc pytest plugin
(src/htr2hpc/pytest_plugin.py), registered via pyproject.toml entry_points.
That plugin uses pytest_load_initial_conftests(tryfirst=True) to run before
pytest-django calls django.setup().
"""
import pytest


@pytest.fixture
def user(db, django_user_model):
    """A basic active user."""
    return django_user_model.objects.create_user(
        username="testuser",
        email="testuser@example.com",
        password="testpass123",
        is_active=True,
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
    mock_instance.find_user.return_value = {"uid": "testnetid"}
    return mock_instance


@pytest.fixture
def mock_user_info_from_ldap(mocker):
    """Patch user_info_from_ldap to avoid real LDAP attribute population."""
    return mocker.patch("htr2hpc.users.user_info_from_ldap")
