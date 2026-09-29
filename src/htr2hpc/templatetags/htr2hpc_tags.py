from datetime import timedelta

from django import template
from django.conf import settings
from django.utils import timezone
from django.utils.timesince import timeuntil

register = template.Library()


@register.simple_tag
def absolute_export_url(domain, export_uri):
    """Absolute URL for an export file.

    Prepends https:// (or http:// in DEBUG) to the bare domain already
    present in the email template context, then appends MEDIA_URL and the
    export path.
    """
    if not domain.startswith(("http://", "https://")):
        scheme = "http" if settings.DEBUG else "https"
        domain = f"{scheme}://{domain}"
    media_url = settings.MEDIA_URL
    return f"{domain.rstrip('/')}{media_url}{export_uri}"


@register.simple_tag
def export_retention_display():
    """Human-readable time until export files expire, using naturaltime.

    Returns empty string when cleanup is disabled (EXPORT_FILE_RETENTION=0).
    """
    hours = getattr(settings, "EXPORT_FILE_RETENTION", 0)
    if not hours:
        return ""
    now = timezone.now()
    expiry = now + timedelta(hours=hours)
    return f"{timeuntil(expiry, now)} from now"
