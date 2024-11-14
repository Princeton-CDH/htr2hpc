from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.executors.threads import ThreadPoolExecutor
from parsl.addresses import address_by_interface

parsl_config = Config(
    executors=[
        ThreadPoolExecutor(max_threads=8, label="local_threads"),
        HighThroughputExecutor(
            label="hpc",
            address=address_by_interface("ib0"),
            cores_per_worker=2,
            provider=SlurmProvider(
                "debug",  # Partition / QOS
                nodes_per_block=2,
                init_blocks=1,
                partition="normal",
                worker_init="module load anaconda3/2024.2; conda activate htr2hpc",
                launcher=SrunLauncher(),
                walltime="00:05:00",
                # Slurm scheduler can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
            ),
        ),
    ],
)
