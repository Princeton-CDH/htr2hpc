# Usage reporting export for staff

Issue: [#216](https://github.com/Princeton-CDH/htr2hpc/issues/216) · PR: [#234](https://github.com/Princeton-CDH/htr2hpc/pull/234)

## Context

Staff need per-user usage numbers for the VMs and for HPC training, broken down by time period, to report on how the service is used. The eScriptorium Leaderboard (`/quotas/instance/`) shows all-time totals only. Its query lives inside the view and can't be reused or extended.

How should htr2hpc give staff a downloadable usage report?

## Decisions

A staff-only CSV export, downloaded from a form at the top of the Leaderboard page.
Each download is one CSV for one report type: total, monthly, or weekly.

### Report contents

- **Report types:**

  - `total`: everything since the selected start date.
  - `month` and `week`: one row per user per *completed* calendar month or Monday-start week since the start date.

- **Start date:** chosen by the user in the form. Defaults to *2026-09-01*, which excludes the beta and summer test data.

- **Same fields for every report type**, so files can be easily combined and processed together:

  `username`, `report_type`, `report_start`, `report_end`, `disk_usage_bytes`, `task_count`, `runtime_seconds`, `cpu_usage`, `vm_segment_tasks`, `vm_transcribe_tasks`, `hpc_segtrain_tasks`, `hpc_train_tasks`

  - `report_start` and `report_end` make each row's period explicit.
  - `report_end` is the last date with activity in the period covered (`Max("started_at")`), for every report type. It comes straight from the query, and the same logic works for totals, months and weeks.
  - The export date is kept in the filename. This preserves when a totals report was downloaded, which can't be inferred from the data.
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
- **View:** a single view at `/usage-report/`, GET only.
  - `report_type` is required (`total`, `month` or `week`).
  - `start` is optional and falls back to the setting below.
  - Invalid parameters return a 400 with the error message.
- **Form:** `UsageReportForm` in `htr2hpc/forms.py`, rendered as one inline row at the top of the Leaderboard.
- **Getting the form into the page:** an inclusion template tag, `{% usage_report_form %}`, called from our Leaderboard template override. eScriptorium's Leaderboard view can't be changed to add the form to its context, so the tag provides it instead.
- **Default start date:** a Django setting, `USAGE_REPORT_DEFAULT_START = date(2026, 9, 1)`, used as the form's initial value.
- htr2hpc URLs live in `htr2hpc_urls.py`, separate from the combined eScriptorium URL config, so tests can use them directly.

## Considered Options

### Where the download is accessible to users

- A form on the existing Leaderboard, at the top of the page.

### Time periods

1. Rolling 7- and 30-day windows. Rejected: the results depend on the day the report is run.
2. Current calendar week and month. Rejected: partial periods unless downloaded on the last day.
3. A ZIP with `totals.csv` plus one `YYYY-MM.csv` per completed month. Rejected:
   - The totals and monthly files had different fields.
   - The date range was only in the filename.
   - It ran one query per month.
4. **One CSV per report type, with explicit date columns and a user-selected start date (chosen).**

## Planned changes

These changes revise the partial implementation in [#234](https://github.com/Princeton-CDH/htr2hpc/pull/234) as of `de45422` (2026-11-04). In that version, the export is a ZIP with a totals CSV and one CSV per completed month, using different fields, a hard-coded start date and hard-coded UTC.

1. **Export logic** (`src/htr2hpc/exports.py`)
   - Merge the totals and monthly field lists, query functions and row functions into one of each.
   - Group by period in the database.
   - Remove the hard-coded start date and the hard-coded UTC.
2. **Setting** (`src/htr2hpc/settings.py`)
   - Add `USAGE_REPORT_DEFAULT_START = date(2026, 9, 1)`.
3. **Form** (`src/htr2hpc/forms.py`, `templates/htr2hpc/usage_report_form.html`)
   - Add `UsageReportForm` with a required `report_type` choice and an optional `start` date, initialized from the setting.
   - Add the form template: one inline row with start date, report type and a Download button.
4. **Template tag** (`src/htr2hpc/templatetags/htr2hpc_tags.py`, `templates/reporting/quotas_leaderboard.html`)
   - Add the `usage_report_form` inclusion tag. Add a code comment explaining that it exists because the form can't be added to eScriptorium's Leaderboard view context.
   - Replace the current button in the Leaderboard override with `{% usage_report_form %}`.
5. **View** (`src/htr2hpc/views.py`)
   - Remove the ZIP and stream a single CSV.
   - Validate the GET parameters with `UsageReportForm`. Return a 400 with the errors if invalid.
6. **Tests** (`tests/test_exports.py`, `tests/test_views.py`)
   - Test against real `TaskReport` fixtures, covering:
     - completed-period boundaries
     - per-job HPC counting
     - users with no activity
     - `report_end` is the last activity date for every report type
   - Check the columns against the shared field list.
   - View: a valid request returns a CSV with the expected filename, and invalid parameters return 400.
   - Remove the `# pragma: no cover` markers.
