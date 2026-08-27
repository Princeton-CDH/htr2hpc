"""Tests for htr2hpc tasks and htr2hpc.train.hpc."""
import sys
from unittest.mock import MagicMock, patch

# htr2hpc.tasks imports from apps.users.consumers (eScriptorium), which is not
# available in the test environment; mock it before importing tasks
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())

from htr2hpc import __version__  # noqa: E402
from htr2hpc.tasks import start_remote_training  # noqa: E402
from htr2hpc.train.hpc import ensure_htr2hpc_version  # noqa: E402


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


class TestStartRemoteTraining:
    def _make_mocks(self):
        user = MagicMock()
        user.username = "testuser"
        user.auth_token.key = "test-token"
        task_report = MagicMock()
        return user, task_report

    @patch("htr2hpc.tasks.send_event")
    @patch("htr2hpc.tasks.ensure_htr2hpc_version", return_value=False)
    @patch("htr2hpc.tasks.Connection")
    def test_version_install_failure_aborts_training(
        self, mock_connection, mock_ensure, mock_send_event
    ):
        """When ensure_htr2hpc_version returns False, training should be aborted."""
        user, task_report = self._make_mocks()
        result = start_remote_training(
            user, "/scratch/working", "train_cmd", 1, 2, task_report
        )
        assert result is False
        # should notify user and record error
        user.notify.assert_called_with(
            "Could not install required htr2hpc version in conda env; aborting training.",
            id="training-error",
            level="danger",
        )
        task_report.error.assert_called_once()
        # should send training:error event
        mock_send_event.assert_called_once_with(
            "document", 1, "training:error", {"id": 2}
        )
        # should not run the training command
        conn = mock_connection.return_value.__enter__.return_value
        assert conn.run.call_count == 0
