from collections import defaultdict

from django.db.models import Q

from htr2hpc.content.models import ContentSnippet


def page_snippets(request):
    """Add live ContentSnippets for the current page to the template context.

    Returns ``page_snippets``, a dict mapping slot names to lists of snippets.
    Each list has site-wide snippets (blank url_path) first, followed by
    snippets for this page. ``url_path`` must match ``request.path`` exactly.

    Only registered in settings when Wagtail (and so this app) is installed.
    """
    snippets = ContentSnippet.objects.filter(live=True).filter(
        Q(url_path=request.path) | Q(url_path="")
    )

    # site-wide (url_path="") sorts before page-specific
    # because the queryset is ordered by url_path ascending ("" < "/...")
    result = defaultdict(list)
    for snippet in snippets:
        result[snippet.slot].append(snippet)

    return {"page_snippets": dict(result)}
