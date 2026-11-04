from typing import ClassVar

from wagtail.snippets.models import register_snippet
from wagtail.snippets.views.snippets import SnippetViewSet

from htr2hpc.content.models import ContentSnippet


class ContentSnippetViewSet(SnippetViewSet):
    """Wagtail admin configuration for ContentSnippet."""

    model = ContentSnippet
    list_display: ClassVar[list] = ["__str__", "slot", "url_path", "is_active"]
    list_filter: ClassVar[list] = ["slot", "is_active"]
    search_fields: ClassVar[list] = ["url_path", "body"]
    ordering: ClassVar[list] = ["url_path", "slot"]


register_snippet(ContentSnippetViewSet)
