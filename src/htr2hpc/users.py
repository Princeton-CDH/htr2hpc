def init_new_user(user, user_info):
    """pucas EXTRA_USER_INIT hook: make new CAS accounts inactive by default."""
    # admins must explicitly activate accounts before users can log in
    user.is_active = False
