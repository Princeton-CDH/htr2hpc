from typing import ClassVar

from django.db import models
from wagtail.fields import RichTextField

SLOT_CHOICES = [
    ("before_main", "Before main content (any page)"),
    ("after_main", "After main content (any page)"),
    ("homepage_heading", "Homepage heading text"),
]


class ContentSnippet(models.Model):
    """Admin-editable rich text content shown on eScriptorium pages.

    Each snippet is identified by a URL path and a slot name. A site-wide
    snippet (blank url_path) and a page-specific snippet can coexist for the
    same slot; the context processor returns both, site-wide first.

    Only one snippet per (url_path, slot) combination is allowed, whether or
    not it is live, so there are no drafts.
    """

    url_path = models.CharField(
        max_length=200,
        blank=True,
        help_text=(
            "URL for the page where this snippet should be displayed (e.g. /contact/). "
            "Leave blank to display on all pages."
        ),
    )
    slot = models.CharField(
        max_length=100,
        choices=SLOT_CHOICES,
        help_text="Where on the page this snippet is displayed.",
    )
    body = RichTextField()
    live = models.BooleanField(
        default=True,
        help_text="Only live snippets are displayed. Uncheck to hide without deleting.",
    )

    class Meta:
        verbose_name = "content snippet"
        verbose_name_plural = "content snippets"
        ordering: ClassVar[list] = ["url_path", "slot"]
        constraints: ClassVar[list] = [
            models.UniqueConstraint(
                fields=["url_path", "slot"],
                name="unique_url_path_slot",
            )
        ]

    def __str__(self):
        scope = self.url_path if self.url_path else "all pages"
        return f"{self.get_slot_display()} — {scope}"
