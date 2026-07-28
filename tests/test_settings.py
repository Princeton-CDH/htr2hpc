"""
Minimal Django settings for running htr2hpc tests.

Does NOT import from htr2hpc.settings or escriptorium.settings —
those pull in PostgreSQL, Redis, Celery broker, and the full eScriptorium
stack. This file defines only what is needed to load the htr2hpc app and
run tests against it.
"""
import os
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
]

DEFAULT_AUTO_FIELD = "django.db.models.AutoField"
SITE_ID = 1

ROOT_URLCONF = "htr2hpc.urls"

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

# pucas LDAP config — minimal; LDAP calls are mocked in tests
PUCAS_LDAP = {
    "ATTRIBUTES": ["givenName", "sn", "mail"],
    "ATTRIBUTE_MAP": {
        "first_name": "givenName",
        "last_name": "sn",
        "email": "mail",
    },
}
