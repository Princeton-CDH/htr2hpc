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
