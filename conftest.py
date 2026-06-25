"""
Root conftest.py — Django setup and module stubs for htr2hpc tests.

users.py imports from users.admin (an eScriptorium module) at the top level.
We stub that module before Django calls setup() so the import resolves without
needing a full eScriptorium checkout.
"""
import sys
import types
from unittest.mock import MagicMock


def pytest_configure(config):
    # Stub users and users.admin before django.setup() so that the
    # module-level "from users.admin import MyUserAdmin" in users.py resolves.
    users_mod = types.ModuleType("users")
    sys.modules.setdefault("users", users_mod)

    admin_mod = types.ModuleType("users.admin")
    # Use Django's built-in UserAdmin as a stand-in for MyUserAdmin.
    # Import is deferred to avoid triggering django.setup() too early.
    from django.contrib.auth.admin import UserAdmin

    admin_mod.MyUserAdmin = UserAdmin
    sys.modules.setdefault("users.admin", admin_mod)

    # pucas.ldap is used at module level in users.py; stub it so tests
    # that don't exercise LDAP don't need a real LDAP server.
    ldap_mod = types.ModuleType("pucas.ldap")
    ldap_mod.LDAPSearch = MagicMock()
    ldap_mod.LDAPSearchException = Exception
    ldap_mod.user_info_from_ldap = MagicMock()
    sys.modules.setdefault("pucas.ldap", ldap_mod)
