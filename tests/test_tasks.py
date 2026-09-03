"""Tests for htr2hpc tasks and htr2hpc.train.hpc."""
import importlib.metadata
import os
import shutil
import subprocess
import sys
from unittest.mock import MagicMock, patch

# htr2hpc.tasks imports from apps.users.consumers (eScriptorium) and celery,
# neither of which is available in the test environment; mock before importing
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("apps", MagicMock())
sys.modules.setdefault("apps.users", MagicMock())
sys.modules.setdefault("apps.users.consumers", MagicMock())

from django.test import override_settings  # noqa: E402
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

    def test_install_command_uses_version_by_default(self):
        """Without HTR2HPC_GITREF override, the command falls back to __version__."""
        conn = MagicMock()
        conn.run.return_value = _mock_run_result(exited=0)
        ensure_htr2hpc_version(conn)
        cmd = conn.run.call_args[0][0]
        assert f"@{__version__}" in cmd
        assert "--upgrade" in cmd

    @override_settings(HTR2HPC_GITREF="abc123sha")
    def test_install_command_uses_gitref_when_set(self):
        """When HTR2HPC_GITREF is set (staging), the command uses it instead of __version__."""
        conn = MagicMock()
        conn.run.return_value = _mock_run_result(exited=0)
        ensure_htr2hpc_version(conn)
        cmd = conn.run.call_args[0][0]
        assert "@abc123sha" in cmd
        assert f"@{__version__}" not in cmd


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


# ---------------------------------------------------------------------------
# integration: verify ensure_htr2hpc_version actually upgrades outdated deps
# ---------------------------------------------------------------------------


def _get_installed_version(package):
    return importlib.metadata.version(package)


def _pip_install(*args):
    # uv-managed venvs do not include pip; use uv pip when available.
    # Pass VIRTUAL_ENV to ensure uv targets the same venv as the running Python.
    if shutil.which("uv"):
        env = {**os.environ, "VIRTUAL_ENV": sys.prefix}
        subprocess.run(["uv", "pip", "install", "-q", *args], check=True, env=env)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


class _LocalHPCConn:
    """Fake Fabric connection that runs pip install locally instead of on HPC.

    ensure_htr2hpc_version constructs a command that uses HPC-specific tools
    (module load, conda run). This stub intercepts that call and runs a local
    equivalent so the function can be tested without an HPC connection.
    """

    def run(self, cmd, warn=False, hide=False):
        # Replace the full HPC command with a local pip install --upgrade.
        # Mirror Fabric's behaviour: always return a result object (never raise),
        # and surface the exit code so ensure_htr2hpc_version can handle failures.
        try:
            _pip_install("--upgrade", ".")
            exited, stderr = 0, ""
        except subprocess.CalledProcessError as e:
            exited, stderr = e.returncode, e.stderr or ""

        class _Result:
            pass

        result = _Result()
        result.exited = exited
        result.stdout = ""
        result.stderr = stderr
        return result


def test_pip_install_uses_pip_when_uv_unavailable():
    """_pip_install falls back to python -m pip when uv is not in PATH."""
    with patch("shutil.which", return_value=None):
        with patch("subprocess.run") as mock_run:
            _pip_install("somepackage==1.0")
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == sys.executable
            assert "pip" in cmd


def test_local_hpc_conn_surfaces_pip_failure():
    """_LocalHPCConn.run returns exited != 0 when pip install fails."""
    with patch(
        "tests.test_tasks._pip_install",
        side_effect=subprocess.CalledProcessError(1, "pip", stderr="error"),
    ):
        conn = _LocalHPCConn()
        result = conn.run("any cmd")
        assert result.exited == 1
        assert result.stderr == "error"


def test_ensure_htr2hpc_version_upgrades_kraken():
    """Verify that ensure_htr2hpc_version upgrades kraken when it has been downgraded.

    Uses a local stub connection that runs pip install --upgrade locally instead
    of on HPC, so the actual function is called end-to-end.
    Uses --no-deps when downgrading kraken to avoid torch dependency conflicts.
    """
    original_kraken = _get_installed_version("kraken")

    try:
        # Simulate a user with kraken 5.x in their conda env.
        # --no-deps avoids torch/torchvision dependency conflicts from kraken 5.x.
        _pip_install("--no-deps", "kraken==5.2.9")
        assert _get_installed_version("kraken") == "5.2.9"

        # Call the actual function under test with a local stub connection
        conn = _LocalHPCConn()
        assert ensure_htr2hpc_version(conn) is True

        # kraken must now satisfy kraken>=6.0
        restored = _get_installed_version("kraken")
        assert int(restored.split(".")[0]) >= 6, (
            f"Expected kraken >= 6.0 after upgrade, got {restored}"
        )
    finally:
        # Restore original kraken so the dev environment is unchanged.
        _pip_install("--no-deps", f"kraken=={original_kraken}")
