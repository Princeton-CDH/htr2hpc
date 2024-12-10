import datetime
import logging
import pathlib
import subprocess

from simple_slurm import Slurm


logger = logging.getLogger(__name__)


def segtrain(
    input_data_dir: pathlib.Path,
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    num_workers: int = 8,
    # optional param to specify name based on document? include date?
) -> int:
    """Run ketos segmentation training as a slurm job.
    Returns the slurm job id for the queued job."""
    segtrain_slurm = Slurm(
        nodes=1,
        ntasks=1,
        cpus_per_task=num_workers,
        mem_per_cpu="4G",
        gres=["gpu:1"],
        job_name="segtrain",
        output=f"segtrain_{Slurm.JOB_ARRAY_MASTER_ID}.out",
        time=datetime.timedelta(minutes=20),
        # time=datetime.timedelta(hours=2),
    )
    # do we want to use CUDA Multi-Process Service (MPS) ?
    # della documentation says to specify with --gpu-mps,
    # but simple slurm doesn't recognize this as a valid directive.
    # Work around that by adding it as a command (if indeed we want this)
    # segtrain_slurm.add_cmd("#SBATCH --gpu-mps")

    # add commands for setup steps
    segtrain_slurm.add_cmd("module purge")
    segtrain_slurm.add_cmd("module load anaconda3/2024.6")
    segtrain_slurm.add_cmd("conda activate htr2hpc")
    logger.debug(f"sbatch file\n: {segtrain_slurm}")
    # sbatch returns the job id for the created job
    segtrain_cmd = (
        # run with default number of epochs (50)
        f"ketos segtrain --resize both -i {input_model}"
        + f" -o {output_model} --workers {num_workers} -d cuda:0 "
        + f"-f xml {input_data_dir}/*.xml "
        # + "--precision 16"  # automatic mixed precision for nvidia gpu
    )

    logger.debug(f"segtrain command: {segtrain_cmd}")
    return segtrain_slurm.sbatch(segtrain_cmd)


def recognition_train(
    input_data_dir: pathlib.Path,
    output_model: pathlib.Path,
    input_model: pathlib.Path = None,
    num_workers: int = 8,
    # optional param to specify name based on document? include date?
) -> int:
    """Run ketos recognition training as a slurm job.
    Returns the slurm job id for the queued job."""
    recogtrain_slurm = Slurm(
        nodes=1,
        ntasks=1,
        cpus_per_task=num_workers,
        mem_per_cpu="2G",
        gres=["gpu:1"],
        job_name="train",
        output=f"train_{Slurm.JOB_ARRAY_MASTER_ID}.out",
        time=datetime.timedelta(minutes=15),
        # time=datetime.timedelta(hours=2),
    )
    recogtrain_slurm.add_cmd("module purge")
    recogtrain_slurm.add_cmd("module load anaconda3/2024.6")
    recogtrain_slurm.add_cmd("conda activate htr2hpc")
    logger.debug(f"sbatch file\n: {recogtrain_slurm}")
    # sbatch returns the job id for the created job

    # input model is optional
    input_model_opt = f"-i {input_model}" if input_model else ""
    recogtrain_cmd = (
        f"ketos train --resize union {input_model_opt}"
        + f" -o {output_model} --workers {num_workers} -d cuda:0 "
        + f"-f binary {input_data_dir}/train.arrow "
    )

    logger.debug(f"recognition train command: {recogtrain_cmd}")
    return recogtrain_slurm.sbatch(recogtrain_cmd)
    # TODO: calling function needs to check for best model
    # or no model improvement


def slurm_job_queue_status(job_id: int) -> str:
    """Use `squeue` to get the full-word status (i.e., PENDING or RUNNING)
    for a queued slurm job."""
    result = subprocess.run(
        ["squeue", f"--jobs={job_id}", "--only-job-state", "--format=%T", "--noheader"],
        capture_output=True,
        text=True,
    )
    # raise subprocess.CalledProcessError if return code indicates an error
    result.check_returncode()
    # return task status without any whitespace
    # squeue doesn't report on the task when it is completed and no longer in the queue,
    # so empty string means the job is complete
    return result.stdout.strip()


def slurm_job_status(job_id: int) -> set:
    """Use `sacct` to get the status of a slurm job that is no longer queued.
    Returns a set of unique full-word statuses, reporting across all tasks for the job.
    """
    result = subprocess.run(
        ["sacct", f"--jobs={job_id}", "--format=state%15", "--noheader"],
        capture_output=True,
        text=True,
    )
    # raise subprocess.CalledProcessError if return code indicates an error
    result.check_returncode()
    # sacct returns a table with status for each portion of the job;
    # return all unique status codes for now
    return set(result.stdout.split())
