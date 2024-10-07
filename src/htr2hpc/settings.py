from escriptorium.settings import INSTALLED_APPS

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

# use local url config
ROOT_URLCONF = "htr2hpc.urls"

# TODO:  put htr2hpc template path in first so it will override
# TEMPLATES = [
#     {
#         'BACKEND': 'django.template.backends.django.DjangoTemplates',
#         'DIRS': [os.path.join(PROJECT_ROOT, 'templates'),
#                  os.path.join(BASE_DIR, 'homepage')],
