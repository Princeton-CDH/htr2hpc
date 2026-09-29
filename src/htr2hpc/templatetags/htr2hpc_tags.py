from django import template
from django.conf import settings

register = template.Library()


@register.simple_tag
def absolute_export_url(domain, export_uri):
    """Absolute URL for an export file.

    Prepends https:// (or http:// in DEBUG) to the bare domain already
    present in the email template context, then appends MEDIA_URL and the
    export path.
    """
    if not domain.startswith("http"):
        scheme = "http" if settings.DEBUG else "https"
        domain = f"{scheme}://{domain}"
    media_url = settings.MEDIA_URL
    return f"{domain.rstrip('/')}{media_url}{export_uri}"


def format_retention(hours):
    """Human-readable retention period, or empty string when cleanup is disabled."""
    if not hours:
        return ""
    if hours % 24 == 0:
        days = hours // 24
        return f"{days} day" if days == 1 else f"{days} days"
    return f"{hours} hour" if hours == 1 else f"{hours} hours"


@register.simple_tag
def export_retention_display():
    """Retention period for user export files, for display in notifications."""
    return format_retention(getattr(settings, "EXPORT_FILE_RETENTION", 0))
