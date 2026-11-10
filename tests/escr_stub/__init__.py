"""Minimal stand-ins for eScriptorium models used by htr2hpc.

eScriptorium isn't installed in the test environment, so these apps define
only the fields htr2hpc queries. Their package names give them the same app
labels as eScriptorium ("reporting", "core"), so ``apps.get_model()`` finds them.

Models live in ``models/__init__.py`` because the repo's test file naming
hook only allows ``test_*.py``, ``__init__.py`` and ``conftest.py`` under tests/.
"""
