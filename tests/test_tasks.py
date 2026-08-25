"""Tests for htr2hpc.train.hpc — ensure_htr2hpc_version."""
from unittest.mock import MagicMock

from htr2hpc import __version__
from htr2hpc.train.hpc import ensure_htr2hpc_version


def _mock_run_result(stdout="", stderr="", exited=0):
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.exited = exited
    return result


class TestEnsureHtr2hpcVersion:
    def test_successful_install_returns_true(self):
        """When pip install succeeds, return True."""
        conn = MagicMock()
        conn.run.return_value = _mock_run_result(exited=0)
        assert ensure_htr2hpc_version(conn) is True
        assert conn.run.call_count == 1

    def test_failed_install_returns_false(self):
        """When pip install fails, return False."""
        conn = MagicMock()
        conn.run.return_value = _mock_run_result(exited=1, stderr="some error")
        assert ensure_htr2hpc_version(conn) is False

    def test_install_command_contains_version(self):
        """The install command should reference the current deployed version."""
        conn = MagicMock()
        conn.run.return_value = _mock_run_result(exited=0)
        ensure_htr2hpc_version(conn)
        cmd = conn.run.call_args[0][0]
        assert f"@v{__version__}" in cmd
