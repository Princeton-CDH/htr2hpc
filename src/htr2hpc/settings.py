from pathlib import Path

from escriptorium.settings import INSTALLED_APPS, TEMPLATES, LOGIN_REDIRECT_URL

# base directory for this package where it is installed
HTR2HPC_INSTALL_DIR = Path(__file__).parent


INSTALLED_APPS += [
    "django_cas_ng",
    "pucas",
]
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

# add custom context processor to display VM status
TEMPLATES[0]["OPTIONS"]["context_processors"].append(
    "htr2hpc.context_processors.vm_status"
)


CUSTOM_HOME = True
