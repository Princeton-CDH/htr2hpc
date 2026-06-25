from types import SimpleNamespace

from htr2hpc.users import init_new_user


def test_init_new_user_sets_inactive():
    user = SimpleNamespace(is_active=True)
    init_new_user(user, {})
    assert user.is_active is False


def test_init_new_user_ignores_user_info():
    user = SimpleNamespace(is_active=True)
    init_new_user(user, {"uid": "netid123", "mail": "netid@example.com"})
    assert user.is_active is False


def test_init_new_user_already_inactive():
    user = SimpleNamespace(is_active=False)
    init_new_user(user, {})
    assert user.is_active is False
