import datetime
import logging
import pathlib
import subprocess

from simple_slurm import Slurm

logger = logging.getLogger(__name__)


def segtrain(
    input_data_dir: pathlib.Path,
    output_model: pathlib.Path,
    input_model: pathlib.Path,
    anaconda_module: str,
    num_workers: int = 8,
    mem_per_cpu: str = "4G",
    training_time: datetime.timedelta = datetime.timedelta(minutes=15),
    epochs: int | None = None,
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
    segtrain_slurm.add_cmd(f"module load {anaconda_module}")
    segtrain_slurm.add_cmd("conda activate htr2hpc")
    # Set a wide virtual terminal width so rich does not truncate metric column
    # labels in the .out file (e.g. "val_accuracy:" instead of "val_accurac…").
    # In a SLURM job, isatty() may return True (pseudo-TTY), causing rich to use
    # the actual terminal width which is often too narrow for all metric columns.
    # rich reads COLUMNS from the environment directly (rich/console.py), so
    # setting it here guarantees enough space regardless of terminal conditions.
    segtrain_slurm.add_cmd("export COLUMNS=200")
    logger.info(f"sbatch file\n: {segtrain_slurm}")
    # sbatch returns the job id for the created job
    segtrain_cmd = (
        f"ketos --workers {num_workers} -d cuda:0 segtrain "
        + f"--min-epochs {epochs} --resize both -i {input_model} -q early -o {output_model} "
        + f"-f xml -t {input_data_dir}/train.txt -e {input_data_dir}/validate.txt "
        # + "--precision 16"  # automatic mixed precision for nvidia gpu
    )

    logger.info(f"segtrain command: {segtrain_cmd}")
    return segtrain_slurm.sbatch(segtrain_cmd)


def recognition_train(
    input_data_dir: pathlib.Path,
    output_model: pathlib.Path,
    input_model: pathlib.Path | None = None,
    anaconda_module: str = "",
    num_workers: int = 8,
    mem_per_cpu: str = "2G",
    training_time: datetime.timedelta = datetime.timedelta(minutes=15),
    epochs: int | None = None,
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
    recogtrain_slurm.add_cmd(f"module load {anaconda_module}")
    recogtrain_slurm.add_cmd("conda activate htr2hpc")
    # Set a wide virtual terminal width so rich does not truncate metric column
    # labels in the .out file (e.g. "val_accuracy:" instead of "val_accurac…").
    # In a SLURM job, isatty() may return True (pseudo-TTY), causing rich to use
    # the actual terminal width which is often too narrow for all metric columns.
    # rich reads COLUMNS from the environment directly (rich/console.py), so
    # setting it here guarantees enough space regardless of terminal conditions.
    recogtrain_slurm.add_cmd("export COLUMNS=200")
    logger.info(f"sbatch file\n: {recogtrain_slurm}")
    # sbatch returns the job id for the created job

    # input model is optional; resize is only used with exesting model
    input_model_opt = f"--resize new -i {input_model}" if input_model else ""
    recogtrain_cmd = (
        f"ketos --workers {num_workers} -d cuda:0 train "
        + f"--min-epochs {epochs} {input_model_opt} -o {output_model} "
        + f"-f binary {input_data_dir}/train.arrow "
        + "-w 0 -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]' -r 0.0001"
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
        check=False,
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
        check=False,
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
        check=False,
    )
    # return task status without any whitespace
    return result.stdout.strip()


def _parse_kb(val_str: str) -> float | None:
    """Parse a memory value string like '12345K', '12M', '2G' into kilobytes.

    Adroit's sacct consistently outputs memory with a 'K' suffix (kilobytes).
    Bare numeric values (no suffix) are treated as kilobytes as a fallback,
    consistent with the most common sacct default. The 'T' suffix is not
    handled as terabyte-scale memory is not expected for training jobs.
    """
    if not val_str:
        return None
    try:
        if val_str.endswith("K"):
            return float(val_str[:-1])
        elif val_str.endswith("M"):
            return float(val_str[:-1]) * 1024
        elif val_str.endswith("G"):
            return float(val_str[:-1]) * 1024 * 1024
        else:
            return float(val_str)
    except ValueError:
        return None


def _parse_memory_output(stdout: str) -> dict:
    """Parse sacct --parsable2 output into max/avg memory in MB, elapsed time, and GPU utilization.

    Expected sacct format: MaxRSS|AveRSS|ElapsedRaw|TRESUsageInAve
    sacct reports memory as e.g. '12345678K', '12M', '2G'.
    TRESUsageInAve contains e.g. 'cpu=...,gres/gpuutil=50' where the GPU
    utilization value is a percentage (0-100); stored as a fraction 0-1.

    Returns dict with max_memory_mb, avg_memory_mb, elapsed_seconds,
    gpu_utilization (all None if unavailable).
    """
    max_kb = None
    avg_kb = None
    elapsed_seconds = None
    gpu_utilization = None

    for line in stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        max_str, avg_str, elapsed_str, tres_str = (
            parts[0].strip(),
            parts[1].strip(),
            parts[2].strip(),
            parts[3].strip(),
        )

        kb = _parse_kb(max_str)
        if kb is not None and (max_kb is None or kb > max_kb):
            max_kb = kb

        kb = _parse_kb(avg_str)
        if kb is not None and avg_kb is None:
            avg_kb = kb

        import contextlib

        if elapsed_str and elapsed_seconds is None:
            with contextlib.suppress(ValueError):
                elapsed_seconds = int(elapsed_str)

        # TRESUsageInAve: "cpu=...,gres/gpuutil=54,..." — GPU utilization is a percentage
        if tres_str and gpu_utilization is None:
            for item in tres_str.split(","):
                if item.startswith("gres/gpuutil="):
                    with contextlib.suppress(ValueError, IndexError):
                        gpu_utilization = float(item.split("=")[1]) / 100

    return {
        "max_memory_mb": round(max_kb / 1024, 2) if max_kb is not None else None,
        "avg_memory_mb": round(avg_kb / 1024, 2) if avg_kb is not None else None,
        "elapsed_seconds": elapsed_seconds,
        "gpu_utilization": gpu_utilization,
    }


def slurm_job_memory(conn, job_id: int) -> dict:
    """Use `sacct` on the HPC host to get resource usage for a completed Slurm job.

    Runs sacct via the existing SSH connection so it executes on the HPC host
    where Slurm is installed.

    Returns a dict with keys:
      - max_memory_mb: peak memory usage in MB (MaxRSS)
      - avg_memory_mb: average memory usage in MB (AveRSS)
      - elapsed_seconds: wall-clock runtime in seconds (ElapsedRaw)
      - gpu_utilization: GPU utilization as fraction 0-1 (from TRESUsageInAve gres/gpuutil)

    Values are None if sacct fails or data is unavailable.
    """
    result = conn.run(
        f"sacct --jobs={job_id} --format=MaxRSS,AveRSS,ElapsedRaw,TRESUsageInAve"
        " --noheader --parsable2",
        warn=True,
        hide=True,
    )
    if result.exited != 0:
        return {
            "max_memory_mb": None,
            "avg_memory_mb": None,
            "elapsed_seconds": None,
            "gpu_utilization": None,
        }
    return _parse_memory_output(result.stdout)
