from typing import Any

from django.contrib.auth.models import AbstractUser


def init_user(user: AbstractUser, user_info: Any) -> None:
    """pucas EXTRA_USER_INIT hook: make new CAS accounts inactive by default.

    Staff and superuser accounts are left active so admin access is not
    interrupted when accounts are re-initialized.
    """
    # admins must explicitly activate accounts before users can log in;
    # skip for staff/superusers so their access is not disrupted
    if not (user.is_staff or user.is_superuser):
        user.is_active = False
