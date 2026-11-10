"""
Minimal Django settings for running htr2hpc tests.

Does NOT import from htr2hpc.settings or escriptorium.settings —
those pull in PostgreSQL, Redis, Celery broker, and the full eScriptorium
stack. This file defines only what is needed to load the htr2hpc app and
run tests against it.
"""

import os
from datetime import date
from pathlib import Path

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
    "htr2hpc",
    # stand-ins for the eScriptorium models htr2hpc queries
    "escr_stub.reporting",
    "escr_stub.core",
]

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
