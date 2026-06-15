import importlib.metadata

from htr2hpc import __version__
from htr2hpc.context_processors import htr2hpc_version


def test_htr2hpc_version():
    context = htr2hpc_version(None)
    assert "HTR2HPC_VERSION" in context
    assert context["HTR2HPC_VERSION"] == __version__
    assert context["HTR2HPC_VERSION"] == importlib.metadata.version("htr2hpc")
