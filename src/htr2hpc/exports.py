"""Data export functions for htr2hpc usage reporting."""

from datetime import UTC, date, datetime

from django.apps import apps
from django.contrib.auth import get_user_model
from django.db.models import (
    Count,
    DurationField,
    ExpressionWrapper,
    F,
    OuterRef,
    Q,
    Subquery,
    Sum,
)

User = get_user_model()

HPC_SEGTRAIN_METHOD = "htr2hpc.tasks.segtrain"
HPC_TRAIN_METHOD = "htr2hpc.tasks.train"
VM_SEGMENT_METHOD = "core.tasks.segment"
VM_TRANSCRIBE_METHOD = "core.tasks.transcribe"

# Earliest date to include in usage reports; excludes beta/summer test data.
USAGE_REPORT_START = date(2026, 9, 1)

TOTALS_FIELDS: list[str] = [
    "username",
    "disk_usage_bytes",
    "total_tasks",
    "total_runtime_seconds",
    "total_cpu_usage",
    "vm_segment_total",
    "vm_transcribe_total",
    "hpc_segtrain_total",
    "hpc_train_total",
]

MONTHLY_FIELDS: list[str] = [
    "username",
    "tasks",
    "runtime_seconds",
    "cpu_usage",
    "vm_segment",
    "vm_transcribe",
    "hpc_segtrain",
    "hpc_train",
]


def _task_filters():
    """Return Q objects for filtering by task method."""
    return {
        "hpc_segtrain": Q(taskreport__method=HPC_SEGTRAIN_METHOD),
        "hpc_train": Q(taskreport__method=HPC_TRAIN_METHOD),
        "vm_segment": Q(taskreport__method=VM_SEGMENT_METHOD),
        "vm_transcribe": Q(taskreport__method=VM_TRANSCRIBE_METHOD),
    }


def _runtime_expr():
    return ExpressionWrapper(
        F("taskreport__done_at") - F("taskreport__started_at"),
        output_field=DurationField(),
    )


def _disk_subquery():
    OcrModel = apps.get_model("core", "OcrModel")
    return (
        OcrModel.objects.filter(owner=OuterRef("pk"))
        .values("owner")
        .annotate(total=Sum("file_size"))
        .values("total")
    )


def get_totals_data():  # pragma: no cover
    """Return annotated user queryset with all-time VM and HPC usage stats.

    Only includes data from USAGE_REPORT_START onwards.
    HPC task counts use distinct task_id to count jobs, not per-page TaskReports.
    """
    start = datetime(
        USAGE_REPORT_START.year,
        USAGE_REPORT_START.month,
        USAGE_REPORT_START.day,
        tzinfo=UTC,
    )
    filter_since_start = Q(taskreport__started_at__gte=start)
    filters = _task_filters()
    runtime = _runtime_expr()

    return (
        User.objects.all()
        .annotate(
            disk_usage=Subquery(_disk_subquery()),
            total_tasks=Count("taskreport", filter=filter_since_start),
            total_runtime=Sum(runtime, filter=filter_since_start),
            total_cpu_usage=Sum("taskreport__cpu_cost", filter=filter_since_start),
            vm_segment_total=Count(
                "taskreport", filter=filters["vm_segment"] & filter_since_start
            ),
            vm_transcribe_total=Count(
                "taskreport",
                filter=filters["vm_transcribe"] & filter_since_start,
            ),
            hpc_segtrain_total=Count(
                "taskreport__task_id",
                filter=filters["hpc_segtrain"] & filter_since_start,
                distinct=True,
            ),
            hpc_train_total=Count(
                "taskreport__task_id",
                filter=filters["hpc_train"] & filter_since_start,
                distinct=True,
            ),
        )
        .order_by("username")
    )


def get_monthly_data(year: int, month: int):  # pragma: no cover
    """Return annotated user queryset with usage stats for one complete month.

    HPC task counts use distinct task_id to count jobs, not per-page TaskReports.
    """
    start = datetime(year, month, 1, tzinfo=UTC)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(year, month + 1, 1, tzinfo=UTC)
    filter_month = Q(
        taskreport__started_at__gte=start,
        taskreport__started_at__lt=end,
    )
    filters = _task_filters()
    runtime = _runtime_expr()

    return (
        User.objects.all()
        .annotate(
            tasks=Count("taskreport", filter=filter_month),
            runtime=Sum(runtime, filter=filter_month),
            cpu_usage=Sum("taskreport__cpu_cost", filter=filter_month),
            vm_segment=Count("taskreport", filter=filters["vm_segment"] & filter_month),
            vm_transcribe=Count(
                "taskreport",
                filter=filters["vm_transcribe"] & filter_month,
            ),
            hpc_segtrain=Count(
                "taskreport__task_id",
                filter=filters["hpc_segtrain"] & filter_month,
                distinct=True,
            ),
            hpc_train=Count(
                "taskreport__task_id",
                filter=filters["hpc_train"] & filter_month,
                distinct=True,
            ),
        )
        .order_by("username")
    )


def get_completed_months() -> list[tuple[int, int]]:
    """Return list of (year, month) tuples for completed months since USAGE_REPORT_START.

    A month is 'complete' if it has fully elapsed — i.e. not the current month.
    """
    today = datetime.now(tz=UTC).date()
    months = []
    year, month = USAGE_REPORT_START.year, USAGE_REPORT_START.month
    while (year, month) < (today.year, today.month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def totals_row(user) -> dict:
    """Convert a totals-annotated user to a CSV row dict."""
    return {
        "username": user.username,
        "disk_usage_bytes": user.disk_usage or 0,
        "total_tasks": user.total_tasks,
        "total_runtime_seconds": int(user.total_runtime.total_seconds())
        if user.total_runtime
        else 0,
        "total_cpu_usage": user.total_cpu_usage or 0,
        "vm_segment_total": user.vm_segment_total,
        "vm_transcribe_total": user.vm_transcribe_total,
        "hpc_segtrain_total": user.hpc_segtrain_total,
        "hpc_train_total": user.hpc_train_total,
    }


def monthly_row(user) -> dict:
    """Convert a monthly-annotated user to a CSV row dict."""
    return {
        "username": user.username,
        "tasks": user.tasks,
        "runtime_seconds": int(user.runtime.total_seconds()) if user.runtime else 0,
        "cpu_usage": user.cpu_usage or 0,
        "vm_segment": user.vm_segment,
        "vm_transcribe": user.vm_transcribe,
        "hpc_segtrain": user.hpc_segtrain,
        "hpc_train": user.hpc_train,
    }


# Keep legacy name for backwards compatibility with existing view import
USAGE_REPORT_FIELDS = TOTALS_FIELDS


def get_usage_data():  # pragma: no cover
    """Alias for get_totals_data() for backwards compatibility."""
    return get_totals_data()


def user_to_csv_row(user) -> dict:
    """Alias for totals_row() for backwards compatibility."""
    return totals_row(user)
