from django.contrib import admin
from django.contrib.auth import get_user_model

from pucas.admin import CasUserAdmin
from users.admin import MyUserAdmin


class Htr2HpcUserAdmin(CasUserAdmin, MyUserAdmin):
    """Extends eScriptorium's UserAdmin with pucas CAS user management."""
    pass


# replace eScriptorium's User registration with our extended admin
admin.site.unregister(get_user_model())
admin.site.register(get_user_model(), Htr2HpcUserAdmin)
