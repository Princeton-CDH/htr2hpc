"""Test setup for loading eScriptorium's real models.

eScriptorium isn't a pip-installable package and its full dependencies (celery,
channels, Elasticsearch...) are too heavy for CI. Instead:

- CI clones eScriptorium (see .github/workflows/unit-tests.yml) and
  ``ESCRIPTORIUM_PATH`` points at that checkout; ``app/`` and ``app/apps/`` are
  added to ``sys.path`` so the ``users``, ``reporting`` and ``core`` apps import
  the same way they do in eScriptorium.
- Small pure-Python dependencies the models subclass or register (Django REST
  framework, ordered_model, solo, easy_thumbnails, django_prometheus) are
  installed without their own dependencies (the ``escriptorium`` dependency group).
- Packages the model modules only import at the top but that aren't needed
  for queries in tests are stubbed in ``sys.modules`` before Django starts.
  This is the same approach used for Pyodide in simulating-risk.

This module is imported at the top of test_settings.py, because pytest-django
calls django.setup() before any conftest.py is loaded.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

ESCRIPTORIUM_PATH = Path(
    os.environ.get("ESCRIPTORIUM_PATH", Path(__file__).parent.parent / "escriptorium")
)

if not (ESCRIPTORIUM_PATH / "app" / "apps").is_dir():
    raise RuntimeError(
        f"eScriptorium checkout not found at {ESCRIPTORIUM_PATH}. "
        "Clone it there, or set ESCRIPTORIUM_PATH to an existing checkout "
        "(see DEVELOPERNOTES.md)."
    )

sys.path[:0] = [str(ESCRIPTORIUM_PATH / "app"), str(ESCRIPTORIUM_PATH / "app" / "apps")]

# imported by eScriptorium (and htr2hpc.tasks) at module level, not used in tests
STUBBED_MODULES = [
    # used by eScriptorium's core template tags, which Django loads with the app
    "bleach",
    "celery",
    "celery.signals",
    "channels",
    "channels.generic",
    "channels.generic.websocket",
    "channels.layers",
    "elasticsearch",
    # escriptorium/__init__.py imports its celery app
    "escriptorium.celery",
    "escriptorium.utils",
    # htr2hpc.tasks imports from eScriptorium's users app as "apps.users"
    "apps",
    "apps.users",
    "apps.users.consumers",
]
for name in STUBBED_MODULES:
    sys.modules.setdefault(name, MagicMock(name=name))
