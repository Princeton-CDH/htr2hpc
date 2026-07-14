"""
Minimal Django settings for running htr2hpc tests.

Does NOT import from htr2hpc.settings or escriptorium.settings —
those pull in PostgreSQL, Redis, Celery broker, and the full eScriptorium
stack. This file defines only what is needed to load the htr2hpc app and
run tests against it.

eScriptorium's app directory must be on PYTHONPATH before running tests;
see DEVELOPERNOTES.md for setup instructions.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# channels is imported by users.consumers at the top level.
# Mock it so we don't need a running ASGI server.
# Note: do NOT mock asgiref — Django itself depends on it.
for _mod in [
    "channels",
    "channels.generic",
    "channels.generic.websocket",
    "channels.layers",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

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
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.messages",
    "django.contrib.sessions",
    "captcha",
    "rest_framework",
    "rest_framework.authtoken",
    "users",
    "htr2hpc",
]

AUTH_USER_MODEL = "users.User"
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

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

# Quota settings referenced by users.models.User methods
QUOTA_DISK_STORAGE = None
QUOTA_CPU_MINUTES = None
QUOTA_GPU_MINUTES = None

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
