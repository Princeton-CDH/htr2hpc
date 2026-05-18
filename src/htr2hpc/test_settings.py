"""
Minimal Django settings for running htr2hpc tests.

Does NOT import from htr2hpc.settings or escriptorium.settings —
those pull in PostgreSQL, Redis, Celery broker, and the full eScriptorium
stack. This file defines only what is needed to load the htr2hpc app and
run tests against it.
"""
import sys
from pathlib import Path

# Put eScriptorium apps on sys.path so imports like
# "from users.admin import MyUserAdmin" resolve.
# pytest.ini_options pythonpath handles this for pytest runs,
# but we also set it here for any direct django-admin invocations.
# Use Path(__file__) without .resolve() to preserve symlink paths.
# htr2hpc may be a symlink inside an escriptorium checkout; resolving
# the symlink would lose the path to the escriptorium root.
_HERE = Path(__file__)

# Search upward for an 'app/apps' directory (eScriptorium root).
def _find_escriptorium_root(start: Path):
    for parent in [start, *start.parents]:
        if (parent / "app" / "apps").is_dir():
            return parent
    return None

_ESCRIPTORIUM_ROOT = _find_escriptorium_root(_HERE)
_SRC = str(_HERE.parents[2])  # .../htr2hpc/src/

_paths_to_add = [_SRC]
if _ESCRIPTORIUM_ROOT:
    _paths_to_add += [
        str(_ESCRIPTORIUM_ROOT / "app" / "apps"),
        str(_ESCRIPTORIUM_ROOT / "app"),
    ]

for _p in _paths_to_add:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# channels is imported by users.consumers at the top level.
# Mock it so we don't need a running ASGI server.
# Note: do NOT mock asgiref — Django itself depends on it.
from unittest.mock import MagicMock  # noqa: E402

for _mod in [
    "channels",
    "channels.generic",
    "channels.generic.websocket",
    "channels.layers",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

SECRET_KEY = "test-secret-key-not-for-production"
DEBUG = True
ALLOWED_HOSTS = ["*"]
SITE_ID = 1

# SQLite in-memory: no PostgreSQL required
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Minimal installed apps — only what htr2hpc actually needs.
# Order matters: users must come before htr2hpc so that
# admin.site.unregister(get_user_model()) in users.py succeeds.
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    "rest_framework",
    "rest_framework.authtoken",
    "captcha",
    "users",
    "reporting",
    "htr2hpc",
]

MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

AUTH_USER_MODEL = "users.User"
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "/"
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

ROOT_URLCONF = "htr2hpc.urls"


class DisableMigrations:
    """Skip all migrations — recreate schema directly from models (faster)."""

    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        return None


MIGRATION_MODULES = DisableMigrations()

# Celery: run tasks synchronously in tests, no broker needed
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Dummy cache — no Redis needed
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.dummy.DummyCache",
    }
}

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

STATIC_URL = "/static/"
MEDIA_ROOT = "/tmp/htr2hpc_test_media"

# Email backend that discards all mail
EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
