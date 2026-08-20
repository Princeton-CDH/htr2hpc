"""Tests for htr2hpc.train.hpc — ensure_kraken_version."""
from unittest.mock import MagicMock

from htr2hpc.train.hpc import REQUIRED_KRAKEN_MAJOR, ensure_kraken_version


def _mock_run_result(stdout="", stderr="", exited=0):
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.exited = exited
    return result


def _make_conn(check_stdout="", check_exited=0):
    """Return a mock fabric Connection whose run() returns the given check result
    on the first call (the version check) and a success result on subsequent calls."""
    conn = MagicMock()
    check_result = _mock_run_result(stdout=check_stdout, exited=check_exited)
    upgrade_result = _mock_run_result(stdout="", exited=0)
    conn.run.side_effect = [check_result, upgrade_result]
    return conn


class TestEnsureKrakenVersion:
    def test_current_version_no_upgrade(self):
        """When the installed major version meets the requirement, return False."""
        conn = _make_conn(check_stdout=f"{REQUIRED_KRAKEN_MAJOR}.0.3\n")
        assert ensure_kraken_version(conn) is False
        # upgrade command should NOT have been called
        assert conn.run.call_count == 1

    def test_newer_version_no_upgrade(self):
        """When the installed major version exceeds the requirement, return False."""
        conn = _make_conn(check_stdout=f"{REQUIRED_KRAKEN_MAJOR + 1}.1.0\n")
        assert ensure_kraken_version(conn) is False
        assert conn.run.call_count == 1

    def test_old_version_triggers_upgrade(self):
        """When the installed major version is below the requirement, upgrade and return True."""
        conn = _make_conn(check_stdout=f"{REQUIRED_KRAKEN_MAJOR - 1}.4.1\n")
        assert ensure_kraken_version(conn) is True
        # both the check and the upgrade command should have been called
        assert conn.run.call_count == 2

    def test_upgrade_command_contains_required_version(self):
        """The upgrade pip install command should pin the required major version."""
        conn = _make_conn(check_stdout=f"{REQUIRED_KRAKEN_MAJOR - 1}.4.1\n")
        ensure_kraken_version(conn)
        upgrade_call_args = conn.run.call_args_list[1]
        cmd = upgrade_call_args[0][0]
        assert f"kraken~={REQUIRED_KRAKEN_MAJOR}.0" in cmd

    def test_check_command_failure_returns_false(self):
        """When the version check command fails, return False without upgrading."""
        conn = _make_conn(check_stdout="", check_exited=1)
        assert ensure_kraken_version(conn) is False
        assert conn.run.call_count == 1

    def test_unparseable_version_returns_false(self):
        """When the version string cannot be parsed, return False without upgrading."""
        conn = _make_conn(check_stdout="not-a-version\n")
        assert ensure_kraken_version(conn) is False
        assert conn.run.call_count == 1

    def test_empty_version_returns_false(self):
        """When the version string is empty, return False without upgrading."""
        conn = _make_conn(check_stdout="\n")
        assert ensure_kraken_version(conn) is False
        assert conn.run.call_count == 1
