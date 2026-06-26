"""
pytest plugin for htr2hpc — registered via pyproject.toml entry_points["pytest11"].

Entry-point plugins load before conftest files and before pytest-django calls
django.setup(), so this is the right place to mock modules that are unavailable
in the test environment.

eScriptorium's app directory must be on PYTHONPATH before running tests.
See DEVELOPERNOTES.md for setup instructions.
"""
import sys
from unittest.mock import MagicMock

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    """Mock unavailable modules before pytest-django calls django.setup()."""
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
