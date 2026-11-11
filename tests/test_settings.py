"""
Minimal Django settings for running htr2hpc tests.

Does NOT import from htr2hpc.settings or escriptorium.settings —
those pull in Redis, the Celery broker, and the full eScriptorium stack. This file defines only what is needed to load the htr2hpc app and
the eScriptorium models it queries (see test_escriptorium_stubs.py).
"""

import os
from datetime import date
from pathlib import Path

# must run before Django loads the eScriptorium apps listed below
import test_escriptorium_stubs  # noqa: F401

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "test-secret-key-for-testing-only")
DEBUG = True
ALLOWED_HOSTS = ["*"]

# PostgreSQL, like eScriptorium (some eScriptorium models use Postgres-only
# fields). Connection settings can be overridden with environment variables;
# pytest-django creates and drops a separate test database.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "htr2hpc"),
        "USER": os.environ.get("POSTGRES_USER", "postgres"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "postgres"),
        "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
    }
}

INSTALLED_APPS = [
    "django.contrib.admin.apps.SimpleAdminConfig",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.messages",
    "django.contrib.sessions",
    "django.contrib.sites",
    # apps whose models eScriptorium's models depend on
    "rest_framework.authtoken",
    "easy_thumbnails",
    # eScriptorium apps with the models htr2hpc queries (see conftest.py)
    "users",
    "versioning",
    "reporting",
    "core",
    "htr2hpc",
]

AUTH_USER_MODEL = "users.User"

DEFAULT_AUTO_FIELD = "django.db.models.AutoField"
SITE_ID = 1

# match eScriptorium's timezone settings
USE_TZ = True
TIME_ZONE = "UTC"

MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "tests.test_urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


class DisableMigrations:
    """Skip all migrations — recreate schema directly from models (faster)."""

    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        return None


MIGRATION_MODULES = DisableMigrations()

# required to load eScriptorium's models: versioning.Versioned uses it as a field default
VERSIONING_DEFAULT_SOURCE = "eScriptorium"

# htr2hpc-specific settings
HPC_HOSTNAME = "della.princeton.edu"
HPC_SSH_KEYFILE = "/tmp/test_ssh_key"
HTR2HPC_INSTALL_DIR = Path(__file__).parent
HPC_ANACONDA_MODULE = "anaconda3/2025.6"
EXPORT_FILE_RETENTION = 168
USAGE_REPORT_DEFAULT_START = date(2026, 9, 1)

# pucas LDAP config — minimal; LDAP calls are mocked in tests
PUCAS_LDAP = {
    "ATTRIBUTES": ["givenName", "sn", "mail"],
    "ATTRIBUTE_MAP": {
        "first_name": "givenName",
        "last_name": "sn",
        "email": "mail",
    },
}
