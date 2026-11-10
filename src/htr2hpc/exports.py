"""Data export functions for htr2hpc usage reporting.

See docs/design-records/216-usage-reporting.md for the design decisions.
"""

from datetime import date, datetime, time, timedelta

from django.apps import apps
from django.db.models import (
    Count,
    DateField,
    DurationField,
    ExpressionWrapper,
    F,
    IntegerField,
    Max,
    OuterRef,
    Q,
    Subquery,
    Sum,
    Value,
)
from django.db.models.functions import Coalesce, TruncDate, TruncMonth, TruncWeek
from django.utils import timezone

HPC_SEGTRAIN_METHOD = "htr2hpc.tasks.segtrain"
HPC_TRAIN_METHOD = "htr2hpc.tasks.train"
VM_SEGMENT_METHOD = "core.tasks.segment"
VM_TRANSCRIBE_METHOD = "core.tasks.transcribe"

REPORT_TOTAL = "total"
REPORT_MONTH = "month"
REPORT_WEEK = "week"

REPORT_TYPE_CHOICES = [
    (REPORT_TOTAL, "Total"),
    (REPORT_MONTH, "Monthly"),
    (REPORT_WEEK, "Weekly"),
]

# report type label used in the download filename
REPORT_FILENAME_LABELS = {
    REPORT_TOTAL: "total",
    REPORT_MONTH: "monthly",
    REPORT_WEEK: "weekly",
}

# Same fields for every report type, so exported files can be combined
USAGE_REPORT_FIELDS: list[str] = [
    "username",
    "report_type",
    "report_start",
    "report_end",
    "disk_usage_bytes",
    "task_count",
    "runtime_seconds",
    "cpu_usage",
    "vm_segment_tasks",
    "vm_transcribe_tasks",
    "hpc_segtrain_tasks",
    "hpc_train_tasks",
]


def report_date_range(
    start: date, report_type: str, today: date
) -> tuple[date, date | None]:
    """Return the first date and the end date (exclusive) of data to include.

    Totals cover everything from the start date on, so there is no end date.
    Monthly and weekly reports only include *completed* periods: from the
    first month (or Monday-start week) beginning on or after the start date,
    up to the start of the current month (or week).
    """
    if report_type == REPORT_MONTH:
        if start.day == 1:
            first = start
        else:
            # first day of the following month
            first = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        return first, today.replace(day=1)
    if report_type == REPORT_WEEK:
        # next Monday, unless the start date is already a Monday
        first = start + timedelta(days=(7 - start.weekday()) % 7)
        return first, today - timedelta(days=today.weekday())
    return start, None


def _start_of_day(day: date) -> datetime:
    """Timezone-aware start of the given day, in the site's configured timezone."""
    return timezone.make_aware(datetime.combine(day, time.min))


def get_usage_report(start: date, report_type: str):
    """Return per-user usage stats as a values queryset, one row per user and period.

    Grouping by month or week is done in the database. Only users with
    activity in a period get a row for that period. ``report_end`` is the
    last date with activity in the period, for every report type.
    HPC task counts use distinct task_id to count jobs, not per-page TaskReports.
    """
    TaskReport = apps.get_model("reporting", "TaskReport")
    first, end = report_date_range(start, report_type, timezone.localdate())

    reports = TaskReport.objects.filter(started_at__gte=_start_of_day(first))
    if end is not None:
        reports = reports.filter(started_at__lt=_start_of_day(end))

    if report_type == REPORT_MONTH:
        report_start = TruncMonth("started_at", output_field=DateField())
    elif report_type == REPORT_WEEK:
        report_start = TruncWeek("started_at", output_field=DateField())
    else:
        report_start = Value(start, output_field=DateField())

    rows = (
        reports.annotate(report_start=report_start)
        .values("user", "user__username", "report_start")
        .annotate(
            report_end=TruncDate(Max("started_at")),
            task_count=Count("pk"),
            runtime=Sum(
                ExpressionWrapper(
                    F("done_at") - F("started_at"), output_field=DurationField()
                )
            ),
            cpu_usage=Sum("cpu_cost"),
            vm_segment_tasks=Count("pk", filter=Q(method=VM_SEGMENT_METHOD)),
            vm_transcribe_tasks=Count("pk", filter=Q(method=VM_TRANSCRIBE_METHOD)),
            hpc_segtrain_tasks=Count(
                "task_id", filter=Q(method=HPC_SEGTRAIN_METHOD), distinct=True
            ),
            hpc_train_tasks=Count(
                "task_id", filter=Q(method=HPC_TRAIN_METHOD), distinct=True
            ),
        )
        .order_by("user__username", "report_start")
    )

    # disk usage is a current snapshot, so it is only reported for totals
    if report_type == REPORT_TOTAL:
        OcrModel = apps.get_model("core", "OcrModel")
        disk_usage = (
            OcrModel.objects.filter(owner=OuterRef("user"))
            .values("owner")
            .annotate(total=Sum("file_size"))
            .values("total")
        )
        rows = rows.annotate(
            disk_usage_bytes=Coalesce(
                Subquery(disk_usage, output_field=IntegerField()), 0
            )
        )
    return rows


def report_row(row: dict, report_type: str) -> dict:
    """Convert one row from :func:`get_usage_report` to a CSV row dict."""
    runtime = row["runtime"]
    return {
        "username": row["user__username"],
        "report_type": report_type,
        "report_start": row["report_start"],
        "report_end": row["report_end"],
        # blank for monthly and weekly reports
        "disk_usage_bytes": row.get("disk_usage_bytes", ""),
        "task_count": row["task_count"],
        "runtime_seconds": int(runtime.total_seconds()) if runtime else 0,
        "cpu_usage": row["cpu_usage"] or 0,
        "vm_segment_tasks": row["vm_segment_tasks"],
        "vm_transcribe_tasks": row["vm_transcribe_tasks"],
        "hpc_segtrain_tasks": row["hpc_segtrain_tasks"],
        "hpc_train_tasks": row["hpc_train_tasks"],
    }
