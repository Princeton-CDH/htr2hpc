# HPC job resource stats for staff

Issue: [#242](https://github.com/Princeton-CDH/htr2hpc/issues/242) · PR: [#244](https://github.com/Princeton-CDH/htr2hpc/pull/244)

## Context

Staff need a way to generate a report of HPC usage by eScriptorium users running training jobs. The usage report from [#216](216-usage-reporting.md) counts training tasks per user, but it doesn't have details on the system resources used for training. Those are only available on the HPC system in the Slurm database, which is currently Adroit.

We asked Princeton Research Computing (RC) whether they have existing tools for collecting job stats across multiple Slurm jobs and users, and RC pointed us to the Jupyter notebook in [`job_defense_shield/scripts`](https://github.com/PrincetonUniversity/job_defense_shield/tree/main/scripts) (`mig_calculator_for_jobstats.ipynb`).

The notebook:

- runs `sacct` with one row per job (`-X`, `-P`)
- counts GPUs from `alloctres` and computes GPU-hours as `elapsedraw` × GPUs
- decodes the **Jobstats** summary in each job's `admincomment` field, to get CPU and GPU memory used and allocated

**Jobstats** is RC's job statistics tool for Slurm. When a job finishes, it stores a summary of the job's CPU and GPU usage in the Slurm database, in the job's administrator comment (`admincomment`). It is the same data RC emails to users after a job (for example "CPU memory usage per node - used/allocated"). The value has the form `JS1:` + base64(gzip(JSON)); `JS1:Short` and `JS1:None` mean no data was recorded.

The notebook is written for interactive analysis with pandas, for a different purpose (deciding whether to convert GPUs to MIG). It also drops jobs with no Jobstats data.

How should htr2hpc turn this into a report that staff can use?

## Decisions

A CSV report with **one row per Slurm job** that ran in a date range, built from `sacct --json`. It is a separate download from the usage report, with its own button on the Leaderboard, so the usage report still works when Adroit is unavailable. The two can be joined on `username`.

### Report contents

- **One row per Slurm job** (`sacct -X`, so job steps are not listed separately). Jobs are not linked to eScriptorium tasks. Most training tasks have two jobs (a calibration job and the training job); the `phase` field distinguishes them.

- **Fields** (proposed by @cmroughan):

  | Field | Meaning | Source |
  | --- | --- | --- |
  | `username` | user's NetID | `user` |
  | `job_id` | Slurm job ID | `job_id` |
  | `train_type` | `segtrain` or `train` | job name |
  | `phase` | `calibration` for an htr2hpc calibration job, `training` for the subsequent training job | job name |
  | `cluster` | Slurm cluster (currently `adroit`); useful when training can be routed to other clusters (e.g. Della) | `cluster` |
  | `job_start` | date and time the job started (ISO 8601) | `time.start` |
  | `time_alloc` | time requested for the job, in seconds | `time.limit` (minutes) |
  | `time_used` | actual run time in seconds; queue time is not included | `time.elapsed` |
  | `exit_code` | Slurm job state, e.g. `COMPLETED`, `TIMEOUT`, `OUT_OF_MEMORY`, `FAILED` | `state.current` |
  | `cpu_hours` | allocated CPU hours: run time × CPU cores | calculated |
  | `gpu_hours` | allocated GPU hours: run time × GPUs | calculated |
  | `cpu_cores` | CPU cores allocated (`--cpus-per-task`) | `tres.allocated` |
  | `cpu_mem_alloc_gb` | CPU memory allocated, in GB | `tres.allocated` |
  | `cpu_mem_used_gb` | CPU memory actually used, in GB | Jobstats |
  | `gpu_mem_alloc_gb` | total memory of the allocated GPU, in GB | Jobstats |
  | `gpu_mem_used_gb` | GPU memory used, in GB | Jobstats |

  htr2hpc jobs always request one node, one task and one GPU, so node, task and GPU counts are not reported. Efficiency and utilization percentages can be calculated from the allocated and used values, so they are not reported either.

- **Jobstats fields are blank when Jobstats has no data.** Short jobs (`JS1:Short`), jobs with no data (`JS1:None`) and unreadable values still get a row, with `cpu_mem_used_gb`, `gpu_mem_alloc_gb` and `gpu_mem_used_gb` left blank. The job still used allocation time, so dropping it would undercount. This differs from RC's notebook, which drops them. Very short jobs can end before the GPU is ever measured, usually because of an error, so this case is expected.

- **GPU fields are blank, not zero,** when there is no GPU data, so they don't lower averages.

- **Job state is reported** so staff can see at a glance whether jobs are failing because they ran out of time or memory.

### Scope of jobs

There is no htr2hpc Slurm account, so jobs can't be selected by account. The first version selects jobs in two steps:

1. Find the users who ran HPC training tasks (`htr2hpc.tasks.segtrain` and `htr2hpc.tasks.train`) in the report period, from the eScriptorium task reports used by the usage report.
2. Query `sacct` for only those users (`-u`, never `-a`), then keep only jobs whose name matches htr2hpc's naming pattern (`[calibrate_]segtrain:<model>` or `[calibrate_]train:<model>`), so a user's other Adroit jobs are not reported.

### Implementation

- **Shared module** `htr2hpc/train/jobstats.py`, with no Django dependency in the core functions so it can be reused (for example by a scheduled job later):
  - `sacct_command` builds the `sacct` command line (`--json`, `-X`, start and end dates, `-u` list).
  - `run_sacct` runs it, either locally or through a `fabric` connection, and returns the JSON text. Tests replace it with recorded output.
  - `parse_sacct_json` turns the JSON into a list of job stat dicts.
  - `decode_admincomment` decodes the Jobstats value.
  - `JOB_STATS_FIELDS` is the single source for the CSV header.
- **Parsing.** With `--json`, the structured fields (state, times, allocated resources) need no text parsing. Only the Jobstats value in `comment.administrator` is decoded: base64, gunzip, JSON, using the keys `used_memory`, `total_memory`, `gpu_used_memory` and `gpu_total_memory` (in bytes) under each node.
- **Triggering from the Leaderboard.** A `staff_member_required`, POST-only view in `htr2hpc_urls.py` reuses the existing `Echo` streaming and a form like the usage report form. The view connects to Adroit as the staff member who clicked, using the `fabric` connection and key that training already uses (`HPC_HOSTNAME`, `HPC_SSH_KEYFILE`). Only `sacct` runs on Adroit; parsing happens on the VM. It is POST because it triggers a remote command.
- **Management command** `hpc_job_stats` runs the same code from the command line (`--start`, `--end`, `--users`, `--output`), for use on a machine where `sacct` is available.
- **Reference.** The decoding follows the format documented by RC's `job_defense_shield` project (notebook and `efficiency.py`). It is a short, independent implementation and the module docstring points to the reference.

### Open question

Is SSH plus `sacct` fast enough to run on demand from a button? If not, getting the job stats and downloading the CSV should be split into separate steps (generate in the background, then download with a GET). To be tested on Adroit.

## Considered Options

### How the report is generated

1. Locally on the VM (the first version of #244). Rejected: the VM has no `sacct`.
2. Staff run RC's notebook as is. Rejected: it needs pandas and Jupyter, drops jobs without Jobstats data, and doesn't produce the per-job fields we asked RC about.
3. A script, custom notebook or OpenOnDemand app run on the HPC. Not chosen: staff would have to log in to Adroit and copy the file off, but it would avoid SSH from the VM.
4. **From the VM, over SSH to Adroit (chosen).** Staff get the report in the same place as the usage report.

### How staff get the report

1. **Leaderboard button, merged into the usage report.**
   - One file for staff.
   - The usage report is per user and per period, so only CPU and GPU hours fit. Memory and per-job detail are lost.
   - Every usage report download needs an SSH call, and the whole report fails when Adroit is down.
2. **Leaderboard button, separate CSV (chosen).**
   - Same place as the usage report.
   - Keeps per-job detail, and the usage report still works without Adroit.
   - Can be joined with the usage report on `username`.
3. **Script on Adroit.**
   - Simplest: no SSH from the VM and no web code.
   - But staff must log in to Adroit and copy the file off.
   - Staff can only see what their own Adroit account is allowed to see.

### How to limit the report to htr2hpc jobs

1. **Filter on the job name pattern after querying by user (chosen).** `sacct --name` only matches exact names, and our job names include the model name, so the list of names would have to be built from the task reports.
2. Build the exact job name list from the task reports. More precise but more complex.
3. Remove the model name from the Slurm job name, leaving four fixed names that could be passed to `--name`. Easier in code, but not chosen: the model name makes it easier to tell jobs apart in myAdroit and OpenOnDemand, especially when a training job fails.
