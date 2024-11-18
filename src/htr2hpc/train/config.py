from parsl.config import Config
from parsl.providers import SlurmProvider, LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.executors.threads import ThreadPoolExecutor
from parsl.addresses import address_by_interface

parsl_config = Config(
    executors=[
        ThreadPoolExecutor(max_threads=8, label="local"),
        HighThroughputExecutor(
            label="hpc",
            address=address_by_interface("ib0"),
            max_workers_per_node=56,
            provider=SlurmProvider(
                nodes_per_block=8,
                init_blocks=1,
                scheduler_options="#SBATCH --gres=gpu:1",
                worker_init="module load anaconda3/2024.6; conda activate htr2hpc",
                launcher=SrunLauncher(),
            ),
        ),
        # HighThroughputExecutor(
        #     label="hpc",
        #     address=address_by_interface("ib0"),
        #     max_workers_per_node=8,
        #     provider=SlurmProvider(
        #         # Princeton HPC instructions say not to specify the partition,
        #         # but to let the slurm scheduler handle that
        #         nodes_per_block=1,
        #         cores_per_node=8,
        #         mem_per_node=3,  # in GB
        #         init_blocks=1,
        #         scheduler_options="#SBATCH --gres=gpu:1",
        #         worker_init="module load anaconda3/2024.2; conda activate htr2hpc",
        #         launcher=SrunLauncher(),
        #         walltime="00:15:00",
        #         # Slurm scheduler can be slow at times,
        #         # increase the command timeouts
        #         cmd_timeout=120,
        #     ),
        # ),
    ],
)
