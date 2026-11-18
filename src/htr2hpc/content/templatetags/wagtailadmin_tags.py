import threading
from unittest.mock import patch

from django.urls import reverse
from wagtail.admin.templatetags import wagtailadmin_tags as original_tags

# Django requires a module-level `register`; reuse wagtail's library so all
# of the original wagtailadmin tags remain available under this name.
register = original_tags.register

UNSUPPORTED_URLS = [
    "wagtailadmin_api:images:listing",
    "wagtailadmin_api:documents:listing",
]

# `patch` temporarily replaces a module-level attribute that every thread in
# the process shares. The lock makes sure only one request does this at a time;
# otherwise one request could restore the original `reverse` while another
# still needs the patched one. (Under Passenger, as deployed, each process
# handles one request at a time, so this is a safeguard for threaded servers.)
_patch_lock = threading.Lock()


def patched_reverse(url, *args, **kwargs):
    # patch reverse so we can bail out on the ones we know will fail
    if url in UNSUPPORTED_URLS:
        return None
    # otherwise, resolve normally
    return reverse(url, *args, **kwargs)


# based on this solution: https://github.com/wagtail/wagtail/issues/14418#issuecomment-6017391902
# this problem is resolved in wagtail 8.1


@register.simple_tag(name="wagtail_config", takes_context=True)
def custom_wagtail_config(context):
    # patch reverse where wagtail looks it up, not in django.urls
    with (
        _patch_lock,
        patch(
            "wagtail.admin.templatetags.wagtailadmin_tags.reverse",
            new=patched_reverse,
        ),
    ):
        config = original_tags.wagtail_config(context)
    # delete the API entries for apps we're not using
    admin_api = config.get("ADMIN_API", {})
    admin_api.pop("DOCUMENTS", None)
    admin_api.pop("IMAGES", None)
    return config
