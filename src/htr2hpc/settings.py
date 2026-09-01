from pathlib import Path

from escriptorium.settings import INSTALLED_APPS, LOGIN_REDIRECT_URL, MIDDLEWARE, TEMPLATES

# base directory for this package where it is installed
HTR2HPC_INSTALL_DIR = Path(__file__).parent


INSTALLED_APPS += ["django_cas_ng", "pucas", "htr2hpc.apps.Htr2HpcConfig"]
AUTHENTICATION_BACKENDS = (
    "django.contrib.auth.backends.ModelBackend",
    "django_cas_ng.backends.CASBackend",
)


# PUCAS configuration for CAS/LDAP login and user provisioning.
# Only includes non-sensitive configurations that do not change
PUCAS_LDAP = {
    # basic user profile attributes
    "ATTRIBUTES": ["givenName", "sn", "mail"],
    "ATTRIBUTE_MAP": {
        "first_name": "givenName",
        "last_name": "sn",
        "email": "mail",
    },
    # new CAS accounts are inactive by default; admins must activate them
    "EXTRA_USER_INIT": "htr2hpc.users.init_user",
}

# default django-cas behavior is to redirect back to the referrer,
# which puts you at the login page; redirect instead to escriptorium default,
# which is currently configured as the projects list page
CAS_REDIRECT_URL = LOGIN_REDIRECT_URL
CAS_IGNORE_REFERER = True

# use local url config
ROOT_URLCONF = "htr2hpc.urls"

# Insert local templates path first so it will take precedence
TEMPLATES[0]["DIRS"].insert(0, HTR2HPC_INSTALL_DIR / "templates")
# NOTE: we may eventually include this package as an installed app,
# in which case custom templates would be picked up via app dirs;
# But to override escriptorium templates, we need to treat it as a
# template directory and put it first in the list.

# add custom context processors to display VM status and htr2hpc version
TEMPLATES[0]["OPTIONS"]["context_processors"].extend([
    "htr2hpc.context_processors.vm_status",
    "htr2hpc.context_processors.htr2hpc_version",
    "htr2hpc.context_processors.kraken_version",
    "htr2hpc.context_processors.site_domain",
])


CUSTOM_HOME = True

# Remove AccountExpiryMiddleware from eScriptorium's default middleware list.
# We use CAS-managed accounts; expiry is handled via pucas + cron, not middleware.
MIDDLEWARE = [m for m in MIDDLEWARE if m != 'escriptorium.middleware.AccountExpiryMiddleware']

# Number of hours to retain user export files before cleanup.
# Set to 0 to disable automatic cleanup entirely.
EXPORT_FILE_RETENTION = 168  # 1 week

# Anaconda module to load on HPC for conda environment management
HPC_ANACONDA_MODULE = "anaconda3/2025.6"
