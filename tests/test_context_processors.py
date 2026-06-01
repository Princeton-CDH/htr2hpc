import importlib.metadata
from unittest.mock import MagicMock

from htr2hpc.context_processors import htr2hpc_version


def test_htr2hpc_version():
    request = MagicMock()
    context = htr2hpc_version(request)
    assert "HTR2HPC_VERSION" in context
    assert context["HTR2HPC_VERSION"] == importlib.metadata.version("htr2hpc")
