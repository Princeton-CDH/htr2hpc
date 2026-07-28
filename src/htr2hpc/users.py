import datetime
from typing import Any

from django.contrib.auth.models import AbstractUser
from django.utils import timezone


def init_user(user: AbstractUser, user_info: Any) -> None:
    """pucas EXTRA_USER_INIT hook: make new CAS accounts inactive by default.

    Existing accounts are left unchanged to preserve any intentional changes
    (e.g. is_active set by an admin). Admin and staff accounts created via
    the createcasuser command are activated by that command after this hook runs.
    """
    # treat accounts created within the last 5 seconds as new
    just_created = (timezone.now() - user.date_joined) < datetime.timedelta(seconds=5)
    if just_created:
        user.is_active = False
