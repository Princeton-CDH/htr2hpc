import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# Mock modules that are unavailable outside the eScriptorium environment
# so autodoc can import htr2hpc.tasks and htr2hpc.views without errors.
for _mod in [
    "apps",
    "apps.users",
    "apps.users.consumers",
    "celery",
    "channels.generic",
    "channels.generic.websocket",
    "channels.layers",
    "coremltools",
    "coremltools.models",
    "coremltools.models.MLModel",
]:
    sys.modules.setdefault(_mod, MagicMock())

import django  # noqa: E402
from django.conf import settings  # noqa: E402

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.auth",
            "django.contrib.contenttypes",
            "django.contrib.sites",
        ],
        DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}},
        SITE_ID=1,
    )
    django.setup()

from htr2hpc import __version__  # noqa: E402

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

project = "htr2hpc"
copyright = "2026, CDH @ Princeton University"
author = "CDH @ Princeton University"

version = __version__
release = __version__

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = True

html_theme = "alabaster"
html_theme_options = {
    "description": "HTR training on HPC for eScriptorium",
    "github_user": "Princeton-CDH",
    "github_repo": "htr2hpc",
    "github_button": True,
    "badge_branch": "main",
}
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "localtoc.html",
        "searchbox.html",
    ],
}
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}
