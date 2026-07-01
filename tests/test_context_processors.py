"""Tests for htr2hpc.context_processors."""
import importlib.metadata
from unittest.mock import MagicMock, patch

import pytest
from django.test import RequestFactory

from htr2hpc import __version__
from htr2hpc.context_processors import htr2hpc_version, vm_status


@pytest.fixture
def rf():
    return RequestFactory()


def _make_mock_memory(total_gb=8, available_gb=4, percent=50.0):
    mem = MagicMock()
    mem.total = total_gb * 1024**3
    mem.available = available_gb * 1024**3
    mem.percent = percent
    return mem


def test_vm_status_returns_expected_keys(rf):
    request = rf.get("/")
    mock_memory = _make_mock_memory()

    with patch("htr2hpc.context_processors.psutil.virtual_memory", return_value=mock_memory):
        with patch("htr2hpc.context_processors.getloadavg", return_value=(1.0, 0.5, 0.3)):
            result = vm_status(request)

    assert "vm_status" in result
    assert "system_memory" in result


def test_vm_status_load_average_values(rf):
    request = rf.get("/")
    mock_memory = _make_mock_memory()

    with patch("htr2hpc.context_processors.psutil.virtual_memory", return_value=mock_memory):
        with patch("htr2hpc.context_processors.getloadavg", return_value=(2.5, 1.2, 0.8)):
            result = vm_status(request)

    load = result["vm_status"]["load_average"]
    assert load["1"] == 2.5
    assert load["5"] == 1.2
    assert load["15"] == 0.8


def test_vm_status_memory_percent(rf):
    request = rf.get("/")
    mock_memory = _make_mock_memory(percent=75.0)

    with patch("htr2hpc.context_processors.psutil.virtual_memory", return_value=mock_memory):
        with patch("htr2hpc.context_processors.getloadavg", return_value=(0.0, 0.0, 0.0)):
            result = vm_status(request)

    assert result["vm_status"]["used_memory"] == 75.0


def test_vm_status_system_memory_is_raw_object(rf):
    """system_memory should be the raw psutil object, not a formatted string."""
    request = rf.get("/")
    mock_memory = _make_mock_memory()

    with patch("htr2hpc.context_processors.psutil.virtual_memory", return_value=mock_memory):
        with patch("htr2hpc.context_processors.getloadavg", return_value=(0.0, 0.0, 0.0)):
            result = vm_status(request)

    assert result["system_memory"] is mock_memory


def test_vm_status_cpu_count_present(rf):
    request = rf.get("/")
    mock_memory = _make_mock_memory()

    with patch("htr2hpc.context_processors.psutil.virtual_memory", return_value=mock_memory):
        with patch("htr2hpc.context_processors.getloadavg", return_value=(0.0, 0.0, 0.0)):
            result = vm_status(request)

    # cpu_count is set at module load time from os.cpu_count(); just verify it's present
    assert "cpu_count" in result["vm_status"]


def test_htr2hpc_version():
    context = htr2hpc_version(None)
    assert "HTR2HPC_VERSION" in context
    assert context["HTR2HPC_VERSION"] == __version__
    assert context["HTR2HPC_VERSION"] == importlib.metadata.version("htr2hpc")
