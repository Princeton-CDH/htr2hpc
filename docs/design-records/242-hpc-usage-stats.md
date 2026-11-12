# HPC job resource stats for staff

Issue: [#242](https://github.com/Princeton-CDH/htr2hpc/issues/242) · PR: [#244](https://github.com/Princeton-CDH/htr2hpc/pull/244)

## Context

Staff need to report how much HPC capacity htr2hpc training uses. The usage report from [#216](216-usage-reporting.md) counts training tasks per user, but it doesn't have real resource numbers. Those only exist in Slurm on Adroit.

We asked Princeton Research Computing whether they have existing tools for collecting job stats across multiple Slurm jobs and users (CSES ticket #68146, October 2026). We said that for each job we want:

- CPU-hours and GPU-hours
- node count
- CPU and GPU memory used
- CPU and GPU memory allocated

RC pointed us to the Jupyter notebook in [`job_defense_shield/scripts`](https://github.com/PrincetonUniversity/job_defense_shield/tree/main/scripts) (`mig_calculator_for_jobstats.ipynb`). The notebook:

- runs `sacct` with one row per job (`-X`, `-P`)
- counts GPUs from `alloctres` and computes GPU-hours as `elapsedraw` × GPUs
- decodes the Jobstats summary in each job's `admincomment` field, using RC's [`efficiency.py`](https://github.com/PrincetonUniversity/job_defense_shield/blob/main/src/job_defense_shield/efficiency.py), to get CPU and GPU memory used and allocated, and GPU utilization

RC also suggested changes to the notebook for our needs:

- add CPU-hours as `elapsedraw` × `ncpus`
- add `nnodes` to the fields
- replace `-a` (all users) with `-u <netids>` or `-A <account>`

The notebook is written for interactive analysis with pandas, for a different purpose (deciding whether to convert GPUs to MIG). It also drops jobs with no Jobstats data.

How should htr2hpc turn this into a per-job report that staff can use?

## Decisions

A per-job CSV of HPC resource stats, built from `sacct` and Jobstats as in RC's notebook.

> **Open question for review: how do staff get the report?**
>
> 1. A **download button on the Leaderboard**, like the usage report. The VM runs `sacct` on Adroit over SSH. Then either:
>    - a. **merge** the HPC numbers into the existing usage report CSV, or
>    - b. offer a **separate** HPC jobs CSV.
> 2. A **script staff run on Adroit** themselves, which writes the CSV there.
>
> See [Considered Options](#how-staff-get-the-report) for the trade-offs. The sections below apply to every option unless marked.

### Report contents

- **One row per job:** `sacct -X`, so job steps are not listed separately.

- **Fields:**

  `job_id`, `username`, `account`, `partition`, `state`, `start`, `end`, `elapsed_seconds`, `nnodes`, `ncpus`, `ngpus`, `cpu_hours`, `gpu_hours`, `cpu_memory_used_gb`, `cpu_memory_alloc_gb`, `gpu_memory_used_gb_max`, `gpu_memory_used_gb_sum`, `gpu_memory_alloc_gb_max`, `gpu_memory_alloc_gb_sum`, `cpu_efficiency_pct`, `gpu_utilization_pct`

  - `cpu_hours` and `gpu_hours` are allocated time: elapsed time × allocated CPUs or GPUs, as RC suggested. This is what the job used up from the allocation, whether or not it was busy.
  - Memory is in GB. Jobstats reports bytes.
  - `cpu_efficiency_pct` is CPU time used ÷ (elapsed × CPUs).
  - `gpu_utilization_pct` is the mean across the job's GPUs.
  - GPU memory is given as both the max for a single GPU and the sum across all GPUs. Max shows whether a job is close to one GPU's limit. Sum shows the total.

- **Jobs without Jobstats data are kept.** Short jobs (`JS1:Short`), jobs with no data (`JS1:None`) and unreadable values still get a row, with the Jobstats fields left blank. The job still used allocation time, so dropping it would undercount. This differs from RC's notebook, which drops them.

- **GPU fields are blank for CPU-only jobs**, not zero, so they don't lower averages.

- **Scope:** the htr2hpc Slurm account (`-A`) and/or a list of NetIDs (`-u`), never `-a`.

- If the merged option (1a) is chosen, the per-user, per-period report only gets totals (`hpc_cpu_hours`, `hpc_gpu_hours`). The per-job fields above aren't included.

### Implementation

- **Shared module** `htr2hpc/train/jobstats.py`, used by every option:
  - `get_sacct_output` builds and runs the `sacct` command. Tests replace it with recorded output.
  - `decode_admincomment` and `parse_job_stats` are plain functions with no Slurm or Django dependency.
  - `JOB_STATS_FIELDS` is the single source for the CSV header.
- **Following RC's notebook:**
  - the same `sacct` fields, plus `nnodes`, `ncpus`, `account` and `partition`
  - GPU count from `alloctres`
  - the same Jobstats keys as `efficiency.py`: `total_time`, `used_memory`, `total_memory`, `gpu_used_memory`, `gpu_total_memory`, `gpu_utilization`
  - plain Python instead of pandas, so no new dependency
- **Option 1 (button):**
  - The view connects to Adroit as the staff member who clicked, using the `fabric` connection and key that training already uses (`HPC_HOSTNAME`, `HPC_SSH_KEYFILE` in `tasks.py`).
  - Only `sacct` runs on Adroit. Parsing happens on the VM.
  - It is a GET-only, `staff_member_required` view in `htr2hpc_urls.py`, reusing the existing `Echo` streaming and the `{% usage_report_form %}` tag.
  - It needs a new `HPC_SLURM_ACCOUNT` setting.
- **Option 2 (script):** a console script, `htr2hpc-jobstats`, in the `htr2hpc` conda environment on Adroit, like `htr2hpc-train`. Its arguments are `--start`, `--end`, `--account`, `--users` and `--output`.
- **Licensing:** `job_defense_shield` is GPL-2.0 and htr2hpc is Apache-2.0. The decoding is written independently from the documented format (`JS1:` + base64 + gzip + JSON). No code is copied, and nothing is downloaded at runtime the way the notebook `wget`s `efficiency.py`. The module docstring credits RC's notebook and `efficiency.py` as the reference.

## Considered Options

### Where `sacct` runs

1. Locally on the VM (the first version of #244). Rejected: the VM has no `sacct`.
2. Staff run RC's notebook as is. Rejected: it needs pandas and Jupyter, drops jobs without Jobstats data, and doesn't produce the per-job fields we asked RC about.
3. **On Adroit, either over SSH from the VM or by staff directly (chosen).** Which one depends on the open question below.

### How staff get the report

*Open: to be decided in review.*

1. **Leaderboard button, merged into the usage report.**
   - One file for staff.
   - The usage report is per user and per period, so only CPU and GPU hours fit. Memory and efficiency per job are lost.
   - Every usage report download needs an SSH call, and the whole report fails when Adroit is down.
2. **Leaderboard button, separate CSV.**
   - Same place as the usage report.
   - Keeps per-job detail, and the usage report still works without Adroit.
   - Can be joined with the usage report on `username`.
3. **Script on Adroit.**
   - Simplest: no SSH from the VM and no web code.
   - But staff must log in to Adroit and copy the file off.
   - Staff can only see what their own Adroit account is allowed to see.

For both button options: the staff member must have completed HPC setup. We also need to check whether one person's account can see other users' jobs with `-A` (Slurm `PrivateData`), or whether a service account is needed.

## Planned changes

These changes revise the draft implementation in [#244](https://github.com/Princeton-CDH/htr2hpc/pull/244) as of `220c71b` (2026-11-12, after merging current `develop`). In that version, `sacct` runs locally on the VM through a Django management command. The staff URL and view from the first draft were dropped in the merge in favour of `develop`'s versions.

1. **Shared module** (`src/htr2hpc/train/jobstats.py`, all options)
   - Fix the `get_sacct_output` docstring: at least one of `usernames` or `account`, not exactly one.
   - Replace the repeated `round(...) if ... is not None else None` with a small helper.
   - Point the module docstring to this design record and to RC's notebook.
2. **Remove what doesn't work** (`src/htr2hpc/management/commands/hpc_job_stats.py`)
   - Delete the management command and its tests.
3. **Delivery: depends on the open question**
   - Option 1a: add `hpc_cpu_hours` and `hpc_gpu_hours` to `USAGE_REPORT_FIELDS` in `exports.py`, filled from `sacct` over SSH in `UsageReportView`.
   - Option 1b:
     - Run `sacct` through a `fabric` connection in `get_sacct_output`.
     - Add `HPC_SLURM_ACCOUNT`.
     - Add a view at `/usage-report/hpc-jobs/` and a second download row in `usage_report_form.html`.
   - Option 2: add `main()` with `argparse`, register `htr2hpc-jobstats` in `pyproject.toml`, and document it in `train/README.md`.
4. **Docs and tests** (`CHANGELOG.md`, `tests/test_jobstats.py`, `tests/test_views.py`)
   - Add a changelog entry under `0.9` for the chosen option.
   - Keep the decoding tests on Jobstats-format input. Add one row of real `sacct -P` output from Adroit as a fixture.
   - Test the chosen delivery option, with `sacct` or the connection mocked. The CSV header should match `JOB_STATS_FIELDS`.
