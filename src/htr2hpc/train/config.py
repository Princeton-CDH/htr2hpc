from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.executors.threads import ThreadPoolExecutor


parsl_config = Config(
    executors=[
        ThreadPoolExecutor(max_threads=8, label="local_threads"),
        HighThroughputExecutor(
            label="hpc",
            # address=address_by_hostname(),
            max_workers_per_node=56,
            provider=SlurmProvider(
                nodes_per_block=128,
                init_blocks=1,
                partition="normal",
                launcher=SrunLauncher(),
            ),
        ),
    ],
)
