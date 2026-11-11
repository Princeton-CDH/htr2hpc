"""
Minimal Django settings for running htr2hpc tests.

Does NOT import from htr2hpc.settings or escriptorium.settings —
those pull in PostgreSQL, Redis, Celery broker, and the full eScriptorium
stack. This file defines only what is needed to load the htr2hpc app and
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

# SQLite in-memory: no PostgreSQL required
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
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

# settings eScriptorium's models read when they are loaded
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
