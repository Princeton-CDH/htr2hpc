"""
Root conftest.py for htr2hpc tests.

Path setup and module mocking are handled by the htr2hpc pytest plugin
(src/htr2hpc/pytest_plugin.py), registered via pyproject.toml entry_points.
That plugin uses pytest_load_initial_conftests(tryfirst=True) to run before
pytest-django calls django.setup().
"""
