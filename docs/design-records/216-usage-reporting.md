# Usage reporting export for staff

- Status: proposed
- Deciders: @rlskoeser, @cmroughan, @tanhaow
- Date: 2026-11-09

Technical Story: [#216](https://github.com/Princeton-CDH/htr2hpc/issues/216), implemented in [#234](https://github.com/Princeton-CDH/htr2hpc/pull/234)

## Context and Problem Statement

Staff need per-user usage numbers for the VMs and for HPC training, broken down by time period, to report on how the service is used. The eScriptorium Leaderboard (`/quotas/instance/`) shows all-time totals only. Its query lives inside the view and can't be reused or extended.

How should htr2hpc give staff a downloadable usage report?

## Decision Drivers

- The data needs to be easy to process: one consistent CSV shape that can be combined across exports.
- Report periods should be complete and not depend on what day the report is downloaded.
- Data from the beta and summer tests should be excluded.
- The query must stay efficient as the number of users grows.
- eScriptorium is upstream software, so all changes stay in htr2hpc.

## Decision Outcome

A staff-only CSV export, downloaded from a form at the top of the Leaderboard page. Each download is one CSV for one report type: total, monthly, or weekly.

### Report contents

- **Report types:**

  - `total`: everything since the selected start date.
  - `month` and `week`: one row per user per *completed* calendar month or Monday-start week since the start date.

- **Start date:** chosen by the user in the form. It defaults to 2026-09-01, which excludes the beta and summer test data.

- **Same fields for every report type**, so files can be combined and processed together:

  `username`, `report_type`, `report_start`, `report_end`, `disk_usage_bytes`, `tasks_count`, `runtime_seconds`, `cpu_usage`, `vm_segment_tasks`, `vm_transcribe_tasks`, `hpc_segtrain_tasks`, `hpc_train_tasks`

  - `report_start` and `report_end` make each row's period explicit. Totals end on the export date; months and weeks end on their last day.
  - `disk_usage_bytes` is a current snapshot, so it is filled in for totals only and left blank for months and weeks.
  - A user with no tasks in a period has no row for that period.

- **HPC training is counted per job:** distinct `task_id`, not one count per page `TaskReport`.

- **Task types left out:**

  - VM training (`vm_train`, `vm_segtrain`): htr2hpc replaces these with the HPC tasks.
  - GPU usage: the VMs have no GPU, and eScriptorium's `gpu_cost` is a calculated cost unit, not real usage.

- **Filename:** `htr-usage-{total|monthly|weekly}-{start}-{export_date}.csv`

### Implementation

- The query logic lives in `htr2hpc/exports.py`, separate from the view, so it can be tested on its own or run from a script later.
- Monthly and weekly periods are grouped in the database with queryset aggregation (`TruncMonth` / `TruncWeek`). Disk usage comes from a `Subquery` to avoid row multiplication.
- Dates use Django's timezone utilities and follow the site's `TIME_ZONE`.
- The CSV is streamed with Django's `Echo` + `StreamingHttpResponse` pattern over a GET request.
- Access is restricted in the view with `staff_member_required`.
- htr2hpc URLs live in `htr2hpc_urls.py`, separate from the combined eScriptorium URL config, so tests can use them directly.

### Positive Consequences

- One report shape for all periods: no remapping when combining exports.
- Rows don't change depending on when the report is downloaded, because only completed periods are included.
- A single query per download, whatever the date range.

### Negative Consequences

- The Leaderboard template override has to be checked when eScriptorium is upgraded.
- Weekly and monthly reports leave `disk_usage_bytes` blank. Disk usage over time is not tracked.

## Considered Options

### Where the download lives

- A separate report page with a POST form. Rejected: it was an extra page and the wrong HTTP method for a download.
- **A form on the existing Leaderboard, at the top of the page (chosen).** Christine confirmed she can find it easily there.

### Time periods

1. Rolling 7- and 30-day windows. Rejected: the results depend on the day the report is run.
2. Current calendar week and month. Rejected: partial periods unless downloaded on the last day.
3. A ZIP with `totals.csv` plus one `YYYY-MM.csv` per completed month. Rejected:
   - The totals and monthly files had different fields.
   - The date range was only in the filename.
   - It ran one query per month.
4. **One CSV per report type, with explicit date columns and a user-selected start date (chosen).**

### Reusing eScriptorium's reporting

- Adapting the Leaderboard query. Not possible: the query is embedded in the upstream view. Making the upstream reporting reusable is listed as a possible future improvement.

## Revision history

Revisions are described against [#234](https://github.com/Princeton-CDH/htr2hpc/pull/234).

| Date | Commit | Change |
|------|--------|--------|
| 2026-10-19 | `be468c0` | First version: separate report page, POST form, rolling 7/30-day windows |
| 2026-10-21 | `704abfd`…`ee95c35` | Changed to GET, streaming response, `exports.py`, disk usage `Subquery`, button on Leaderboard, calendar week/month |
| 2026-10-28 | `3bbe94f`…`15f3d83` | Split view and export tests, separate `htr2hpc_urls.py`, `Echo` CSV pattern, button at top, staff check in the view |
| 2026-11-04 | `2453ad7`, `de45422` | Christine's feedback: HPC counted per job, VM training and GPU fields dropped, ZIP of totals + completed months, Sept 1 start |
| 2026-11-09 | *planned, from `de45422`* | Rebecca's 11/4 review and the 11/9 Iteration Planning: ZIP replaced by one CSV per report type, shared fields with `report_type` / `report_start` / `report_end`, user-selected start date, weekly report, database aggregation, timezone-aware dates |

### Planned changes from `de45422` (2026-11-04)

1. **Export logic** (`src/htr2hpc/exports.py`)
   - Merge the totals and monthly field lists, query functions and row functions into one of each.
   - Group by period in the database.
   - Remove the hard-coded start date and the hard-coded UTC.
2. **View and download** (`src/htr2hpc/views.py`, `templates/reporting/quotas_leaderboard.html`)
   - Remove the ZIP and stream a single CSV.
   - Replace the button with a form: start date, report type, and a Download button.
3. **Tests** (`tests/test_exports.py`, `tests/test_views.py`)
   - Test against real `TaskReport` fixtures, covering:
     - completed-period boundaries
     - per-job HPC counting
     - users with no activity
   - Check the columns against the shared field list.
   - Remove the `# pragma: no cover` markers.

## Links

- Requirements and decisions: [#216](https://github.com/Princeton-CDH/htr2hpc/issues/216), including the [11/9 Iteration Planning summary](https://github.com/Princeton-CDH/htr2hpc/issues/216#issuecomment-6748295332)
- Per-job HPC resource stats (memory, CPU) are handled separately in [#244](https://github.com/Princeton-CDH/htr2hpc/pull/244)
- Format adapted from [cdh-ansible architecture decisions](https://github.com/Princeton-CDH/cdh-ansible/tree/main/architecture-decisions)
