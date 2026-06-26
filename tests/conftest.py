"""
Pytest fixtures shared across all htr2hpc test modules.

Path setup and module mocks are handled in the root conftest.py via
pytest_configure, which fires before pytest-django calls django.setup().
"""

import pytest


@pytest.fixture
def api_client_instance():
    """An eScriptoriumAPIClient pointed at a fake base URL."""
    from htr2hpc.api_client import eScriptoriumAPIClient

    return eScriptoriumAPIClient(
        base_url="https://escriptorium.example.com",
        api_token="test-token-abc123",
    )
