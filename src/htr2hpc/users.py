import datetime
from typing import Any

from django.contrib.auth.models import AbstractUser
from django.utils import timezone


def init_user(user: AbstractUser, user_info: Any) -> None:
    """pucas EXTRA_USER_INIT hook: make new CAS accounts inactive by default.

    Staff and superuser accounts are left active so admin access is not
    interrupted when accounts are re-initialized. Existing regular user
    accounts are also left unchanged to preserve any intentional changes
    (e.g. is_active set by an admin).
    """
    if user.is_staff or user.is_superuser:
        return
    # treat accounts created within the last 5 seconds as new
    just_created = (timezone.now() - user.date_joined) < datetime.timedelta(seconds=5)
    if just_created:
        user.is_active = False
