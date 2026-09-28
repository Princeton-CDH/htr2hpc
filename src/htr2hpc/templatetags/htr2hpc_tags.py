from django import template
from django.conf import settings

register = template.Library()


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
