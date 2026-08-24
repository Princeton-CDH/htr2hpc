"""Tests for htr2hpc.settings."""
import sys
from types import ModuleType
from unittest.mock import patch

import pytest


def _make_escriptorium_settings(middleware):
    """Create a minimal fake escriptorium.settings module with the given MIDDLEWARE list."""
    escriptorium = ModuleType("escriptorium")
    settings = ModuleType("escriptorium.settings")
    settings.MIDDLEWARE = middleware
    settings.INSTALLED_APPS = []
    settings.LOGIN_REDIRECT_URL = "/"
    settings.TEMPLATES = [{"DIRS": [], "OPTIONS": {"context_processors": []}}]
    escriptorium.settings = settings
    return escriptorium, settings


@pytest.fixture(autouse=True)
def clear_htr2hpc_settings():
    """Remove htr2hpc.settings from sys.modules before and after each test
    so the module-level code re-executes with whatever mock is in place."""
    sys.modules.pop("htr2hpc.settings", None)
    yield
    sys.modules.pop("htr2hpc.settings", None)


def test_middleware_excludes_account_expiry():
    """AccountExpiryMiddleware should be removed from the middleware list."""
    sample_middleware = [
        "django_prometheus.middleware.PrometheusBeforeMiddleware",
        "django.middleware.security.SecurityMiddleware",
        "django.contrib.sessions.middleware.SessionMiddleware",
        "escriptorium.middleware.AccountExpiryMiddleware",
        "django_prometheus.middleware.PrometheusAfterMiddleware",
    ]
    escriptorium, settings = _make_escriptorium_settings(sample_middleware)
    with patch.dict(sys.modules, {"escriptorium": escriptorium, "escriptorium.settings": settings}):
        import htr2hpc.settings as htr2hpc_settings

        assert "escriptorium.middleware.AccountExpiryMiddleware" not in htr2hpc_settings.MIDDLEWARE


def test_middleware_retains_other_entries():
    """All middleware except AccountExpiryMiddleware should be retained."""
    sample_middleware = [
        "django_prometheus.middleware.PrometheusBeforeMiddleware",
        "django.middleware.security.SecurityMiddleware",
        "escriptorium.middleware.AccountExpiryMiddleware",
        "django_prometheus.middleware.PrometheusAfterMiddleware",
    ]
    escriptorium, settings = _make_escriptorium_settings(sample_middleware)
    with patch.dict(sys.modules, {"escriptorium": escriptorium, "escriptorium.settings": settings}):
        import htr2hpc.settings as htr2hpc_settings

        expected = [m for m in sample_middleware if m != "escriptorium.middleware.AccountExpiryMiddleware"]
        assert htr2hpc_settings.MIDDLEWARE == expected
