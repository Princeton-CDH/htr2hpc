from collections import defaultdict

from django.db.models import Q


def page_snippets(request):
    """Inject active ContentSnippets into template context, keyed by slot name.

    Returns a dict mapping slot names to lists of snippets. Each list contains
    site-wide snippets (blank url_path) first, followed by page-specific snippets,
    so site-wide content always renders before page-specific content.

    Returns an empty dict if htr2hpc.content is not installed.
    """
    try:
        from htr2hpc.content.models import ContentSnippet
    except ImportError:
        return {"page_snippets": {}}

    snippets = ContentSnippet.objects.filter(is_active=True).filter(
        Q(url_path=request.path) | Q(url_path="")
    )

    # Group by slot; site-wide (url_path="") sorts before page-specific
    # because the queryset is ordered by url_path ascending ("" < "/...")
    result = defaultdict(list)
    for snippet in snippets:
        result[snippet.slot].append(snippet)

    return {"page_snippets": dict(result)}
