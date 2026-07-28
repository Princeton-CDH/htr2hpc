from django.contrib import admin  # pragma: no cover
from django.contrib.auth import get_user_model  # pragma: no cover

from pucas.admin import CasUserAdmin  # pragma: no cover
from users.admin import MyUserAdmin  # pragma: no cover


class Htr2HpcUserAdmin(CasUserAdmin, MyUserAdmin):  # pragma: no cover
    """Extends eScriptorium's UserAdmin with pucas CAS user management.

    Not covered by tests because this module depends on eScriptorium's MyUserAdmin
    which is not available in the test environment. If functionality is added
    here, update test settings to support it.
    """
    pass


# replace eScriptorium's User registration with our extended admin
admin.site.unregister(get_user_model())  # pragma: no cover
admin.site.register(get_user_model(), Htr2HpcUserAdmin)  # pragma: no cover
