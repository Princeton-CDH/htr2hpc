"""Data export functions for htr2hpc usage reporting.

See docs/design-records/216-usage-reporting.md for the design decisions.
"""

from datetime import date, timedelta

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

# report type label used in the download filename, e.g. "monthly"
REPORT_FILENAME_LABELS = {value: label.lower() for value, label in REPORT_TYPE_CHOICES}

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

    Monthly and weekly reports only include *completed* periods, so the
    requested start date is adjusted:

    - the first period is the first month (or week) that begins on or after
      the start date, so a partial first period is skipped
    - the end date is the start of the current month (or week), so the
      current, incomplete period is excluded

    Weeks start on Monday.
    """
    # no date adjustments are needed when reporting totals; bail out with requested start and no end date
    if report_type == REPORT_TOTAL:
        return start, None
    if report_type == REPORT_MONTH:
        if start.day == 1:
            first = start
        else:
            # first day of the following month
            first = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        return first, today.replace(day=1)
    # weekly: next Monday, unless the start date is already a Monday
    first = start + timedelta(days=(7 - start.weekday()) % 7)
    return first, today - timedelta(days=today.weekday())


def get_usage_report(start: date, report_type: str):
    """Return per-user usage stats as a values queryset, one row per user and period.

    Grouping by month or week is done in the database. Only users with
    activity in a period get a row for that period. ``report_end`` is the
    last date with activity in the period, for every report type.
    Counts, runtime and cpu usage are per job. eScriptorium creates one
    TaskReport per page for a single job, all with the same task_id and
    (nearly) the same times and cpu cost, so summing every report would
    massively over-count. Only the first report of each job is used.
    Reports without a task_id are not Celery tasks and are not included.
    """
    TaskReport = apps.get_model("reporting", "TaskReport")
    first, end = report_date_range(start, report_type, timezone.localdate())

    # Keep the first report (lowest pk) of each job. This is done before the date
    # filters so each job falls in a single period, based on its first report.
    #
    # This has to be done in two passes: Django does not allow annotate() (used
    # below for the per-user aggregates) together with distinct("task_id"), so
    # the de-duplication is a subquery and the aggregation is the outer query.
    # Django inlines it as SQL (`id IN (SELECT DISTINCT ON ...)`), so the ids
    # are never loaded into Python.
    first_report_ids = Subquery(
        TaskReport.objects.exclude(task_id__isnull=True)
        .exclude(task_id="")
        .order_by("task_id", "pk")
        .distinct("task_id")
        .values("pk")
    )
    reports = TaskReport.objects.filter(pk__in=first_report_ids)

    # filter on the date part of started_at, in the site's configured timezone
    reports = reports.filter(started_at__date__gte=first)
    if end is not None:
        reports = reports.filter(started_at__date__lt=end)

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
            hpc_segtrain_tasks=Count("pk", filter=Q(method=HPC_SEGTRAIN_METHOD)),
            hpc_train_tasks=Count("pk", filter=Q(method=HPC_TRAIN_METHOD)),
        )
        .order_by("user__username", "report_start")
    )

    # disk usage is a current snapshot, so it is only reported for totals
    if report_type == REPORT_TOTAL:
        OcrModel = apps.get_model("core", "OcrModel")
        DocumentPart = apps.get_model("core", "DocumentPart")
        # sum model files and document images in separate subqueries so neither
        # source multiplies the other; this matches the Leaderboard's disk usage
        models_size = (
            OcrModel.objects.filter(owner=OuterRef("user"))
            .values("owner")
            .annotate(total=Sum("file_size"))
            .values("total")
        )
        images_size = (
            DocumentPart.objects.filter(document__owner=OuterRef("user"))
            .values("document__owner")
            .annotate(total=Sum("image_file_size"))
            .values("total")
        )
        rows = rows.annotate(
            disk_usage_bytes=Coalesce(
                Subquery(models_size, output_field=IntegerField()), 0
            )
            + Coalesce(Subquery(images_size, output_field=IntegerField()), 0)
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
