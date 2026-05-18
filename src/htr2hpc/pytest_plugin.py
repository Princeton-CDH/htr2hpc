"""
pytest plugin for htr2hpc — registered via pyproject.toml entry_points["pytest11"].

Entry-point plugins load before conftest files and before pytest-django calls
django.setup(), so this is the right place to add eScriptorium to sys.path and
mock modules that are unavailable in the test environment.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock


def _find_escriptorium_root() -> Path | None:
    """Locate the eScriptorium root directory.

    Checks (in order):
    1. ESCRIPTORIUM_ROOT environment variable.
    2. Walk upward from this file's *unresolved* path (preserves symlinks).
    3. Walk upward from this file's *resolved* path (symlink target).
    4. Check for a sibling directory named 'escriptorium' next to the
       resolved htr2hpc checkout — handles the common layout where
       htr2hpc is a symlink inside escriptorium/ but Python resolves it
       to a standalone checkout directory.
    """
    env_root = os.environ.get("ESCRIPTORIUM_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "app" / "apps").is_dir():
            return p

    # Try unresolved path first (works when htr2hpc is symlinked inside escriptorium/)
    here_unresolved = Path(__file__)
    for candidate in [here_unresolved, *here_unresolved.parents]:
        if (candidate / "app" / "apps").is_dir():
            return candidate

    # Try resolved path and its siblings (works when htr2hpc is a standalone checkout
    # that lives next to an escriptorium directory)
    here_resolved = Path(__file__).resolve()
    for candidate in [here_resolved, *here_resolved.parents]:
        if (candidate / "app" / "apps").is_dir():
            return candidate
        # Check for a sibling named 'escriptorium'
        sibling = candidate.parent / "escriptorium"
        if (sibling / "app" / "apps").is_dir():
            return sibling

    return None


import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    """Add eScriptorium apps to sys.path and mock unavailable modules.

    Uses pytest_load_initial_conftests with tryfirst=True so this runs
    before pytest-django calls django.setup() in its own
    pytest_load_initial_conftests hook.
    """
    escriptorium_root = _find_escriptorium_root()
    if escriptorium_root:
        for _p in [
            str(escriptorium_root / "app" / "apps"),
            str(escriptorium_root / "app"),
        ]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

    # channels is imported by users.consumers at the top level.
    # Mock it so we don't need a running ASGI server.
    # Note: do NOT mock asgiref — Django itself depends on it.
    for _mod in [
        "channels",
        "channels.generic",
        "channels.generic.websocket",
        "channels.layers",
    ]:
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock()

    # coremltools has platform-specific binaries not available in CI.
    # Mock it before api_client.py is imported.
    for _mod in ["coremltools", "coremltools.models", "coremltools.models.MLModel"]:
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock()


def pytest_configure(config):
    pass  # kept for compatibility; actual setup is in pytest_load_initial_conftests
