from typing import ClassVar

from django.db import models
from wagtail.fields import RichTextField

SLOT_CHOICES = [
    ("intro", "Page intro"),
    ("announcement", "Announcement banner"),
    ("before_body", "Before body (site-wide)"),
    ("after_body", "After body (site-wide)"),
]


class ContentSnippet(models.Model):
    """Admin-editable rich text content injected into eScriptorium pages.

    Each snippet is identified by a URL path and a slot name. A site-wide
    snippet (blank url_path) and a page-specific snippet can coexist for the
    same slot — the context processor returns both, site-wide first.

    Only one snippet per (url_path, slot) combination is allowed.
    """

    url_path = models.CharField(
        max_length=200,
        blank=True,
        help_text=(
            "URL path this snippet applies to (e.g. /contact/). "
            "Leave blank for site-wide snippets (before_body, after_body)."
        ),
    )
    slot = models.CharField(
        max_length=100,
        choices=SLOT_CHOICES,
        help_text="Where this snippet is injected on the page.",
    )
    body = RichTextField()
    is_active = models.BooleanField(
        default=False,
        help_text="Only active snippets are displayed. Toggle to publish or unpublish.",
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
        scope = self.url_path if self.url_path else "site-wide"
        return f"{self.get_slot_display()} — {scope}"
