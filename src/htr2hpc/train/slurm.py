import datetime
import logging
import pathlib
import subprocess

from simple_slurm import Slurm

from htr2hpc.train.data import TrainingDataCounts

logger = logging.getLogger(__name__)


def segtrain(
    input_data_dir: pathlib.Path,
    output_model: pathlib.Path,
    input_model: pathlib.Path,
    num_workers: int = 8,
    mem_per_cpu: str = "4G",
    training_time: datetime.timedelta = datetime.timedelta(minutes=15),
    epochs: int = None,
    # optional param to specify name based on document? include date?
) -> int:
    """Run ketos segmentation training as a slurm job.
    Returns the slurm job id for the queued job."""

    # no epochs are passed for prelim train task.
    if not epochs:
        epochs = 50
        prelim_opt = "calibrate_"
    else:
        prelim_opt = ""
        
    segtrain_slurm = Slurm(
        nodes=1,
        ntasks=1,
        cpus_per_task=num_workers,
        mem_per_cpu=mem_per_cpu,
        gres=["gpu:1"],
        job_name=f"{prelim_opt}segtrain:{output_model.name}",
        output=f"segtrain_{Slurm.JOB_ARRAY_MASTER_ID}.out",
        time=training_time,
    )
    # do we want to use CUDA Multi-Process Service (MPS) ?
    # della documentation says to specify with --gpu-mps,
    # but simple slurm doesn't recognize this as a valid directive.
    # Work around that by adding it as a command (if indeed we want this)
    # segtrain_slurm.add_cmd("#SBATCH --gpu-mps")

    # add commands for setup steps
    segtrain_slurm.add_cmd("module purge")
    segtrain_slurm.add_cmd("module load anaconda3/2024.2")
    segtrain_slurm.add_cmd("conda activate htr2hpc")
    logger.info(f"sbatch file\n: {segtrain_slurm}")
    # sbatch returns the job id for the created job
    segtrain_cmd = (
        f"ketos segtrain --min-epochs {epochs} --resize both -i {input_model} -q early"
        + f" -o {output_model} --workers {num_workers} -d cuda:0 "
        + f"-f xml -t {input_data_dir}/train.txt -e {input_data_dir}/validate.txt"
        # + "--precision 16"  # automatic mixed precision for nvidia gpu
    )

    logger.info(f"segtrain command: {segtrain_cmd}")
    return segtrain_slurm.sbatch(segtrain_cmd)


def recognition_train(
    input_data_dir: pathlib.Path,
    output_model: pathlib.Path,
    input_model: pathlib.Path = None,
    num_workers: int = 8,
    mem_per_cpu: str = "2G",
    training_time: datetime.timedelta = datetime.timedelta(minutes=15),
    epochs: int = None,
    # optional param to specify name based on document? include date?
) -> int:
    """Run ketos recognition training as a slurm job.
    Returns the slurm job id for the queued job."""
    
    # no epochs are passed for prelim train task.
    if not epochs:
        epochs = 50
        prelim_opt = "calibrate_"
    else:
        prelim_opt = ""

    recogtrain_slurm = Slurm(
        nodes=1,
        ntasks=1,
        cpus_per_task=num_workers,
        mem_per_cpu=mem_per_cpu,
        gres=["gpu:1"],
        job_name=f"{prelim_opt}train:{output_model.name}",
        output=f"train_{Slurm.JOB_ARRAY_MASTER_ID}.out",
        time=training_time,
    )
    recogtrain_slurm.add_cmd("module purge")
    recogtrain_slurm.add_cmd("module load anaconda3/2024.2")
    recogtrain_slurm.add_cmd("conda activate htr2hpc")
    logger.info(f"sbatch file\n: {recogtrain_slurm}")
    # sbatch returns the job id for the created job

    # input model is optional; resize is only used with exesting model
    input_model_opt = f"--resize new -i {input_model}" if input_model else ""
    recogtrain_cmd = (
        f"ketos train --min-epochs {epochs} {input_model_opt}"
        + f" -o {output_model} --workers {num_workers} -d cuda:0 "
        + f"-f binary {input_data_dir}/train.arrow "
        + f"-w 0 -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]' -r 0.0001"
    )

    logger.info(f"recognition train command: {recogtrain_cmd}")
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
    
def slurm_job_stats(job_id: int) -> str:
    """Use `jobstats` to get Slurm Job Statistics, to track resource usage"""
    result = subprocess.run(
        ["jobstats", str(job_id)],
        capture_output=True,
        text=True,
    )
    # return task status without any whitespace
    return result.stdout.strip()
