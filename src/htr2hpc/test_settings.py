"""Minimal Django settings for running htr2hpc tests."""

SECRET_KEY = "test-secret-key-for-testing-only"  # noqa: S105

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
