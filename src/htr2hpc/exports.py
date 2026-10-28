"""Data export functions for htr2hpc usage reporting."""

from datetime import UTC, datetime, timedelta

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
VM_SEGTRAIN_METHOD = "core.tasks.segtrain"
VM_TRAIN_METHOD = "core.tasks.train"

USAGE_REPORT_FIELDS = [
    "username",
    "disk_usage_bytes",
    "total_tasks",
    "this_week_tasks",
    "this_month_tasks",
    "total_runtime_seconds",
    "this_week_runtime_seconds",
    "this_month_runtime_seconds",
    "total_cpu_usage",
    "this_week_cpu_usage",
    "this_month_cpu_usage",
    "total_gpu_usage",
    "this_week_gpu_usage",
    "this_month_gpu_usage",
    "vm_segment_total",
    "vm_segment_this_week",
    "vm_segment_this_month",
    "vm_transcribe_total",
    "vm_transcribe_this_week",
    "vm_transcribe_this_month",
    "vm_segtrain_total",
    "vm_segtrain_this_week",
    "vm_segtrain_this_month",
    "vm_train_total",
    "vm_train_this_week",
    "vm_train_this_month",
    "hpc_segtrain_total",
    "hpc_segtrain_this_week",
    "hpc_segtrain_this_month",
    "hpc_train_total",
    "hpc_train_this_week",
    "hpc_train_this_month",
]


def get_usage_data():  # pragma: no cover
    """Return annotated user queryset with VM and HPC usage stats.

    Uses a Subquery for disk_usage to avoid row multiplication from
    joining both ocrmodel and taskreport relations simultaneously.
    """
    now = datetime.now(tz=UTC)
    # Use calendar month (1st of current month) and fixed week (Monday)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start_of_week = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    filter_this_week = Q(taskreport__started_at__gte=start_of_week)
    filter_this_month = Q(taskreport__started_at__gte=start_of_month)
    filter_hpc_segtrain = Q(taskreport__method=HPC_SEGTRAIN_METHOD)
    filter_hpc_train = Q(taskreport__method=HPC_TRAIN_METHOD)
    filter_vm_segment = Q(taskreport__method=VM_SEGMENT_METHOD)
    filter_vm_transcribe = Q(taskreport__method=VM_TRANSCRIBE_METHOD)
    filter_vm_segtrain = Q(taskreport__method=VM_SEGTRAIN_METHOD)
    filter_vm_train = Q(taskreport__method=VM_TRAIN_METHOD)
    runtime = ExpressionWrapper(
        F("taskreport__done_at") - F("taskreport__started_at"),
        output_field=DurationField(),
    )

    OcrModel = apps.get_model("core", "OcrModel")
    disk_subquery = (
        OcrModel.objects.filter(owner=OuterRef("pk"))
        .values("owner")
        .annotate(total=Sum("file_size"))
        .values("total")
    )

    return (
        User.objects.all()
        .annotate(
            disk_usage=Subquery(disk_subquery),
            total_tasks=Count("taskreport"),
            this_week_tasks=Count("taskreport", filter=filter_this_week),
            this_month_tasks=Count("taskreport", filter=filter_this_month),
            total_runtime=Sum(runtime),
            this_week_runtime=Sum(runtime, filter=filter_this_week),
            this_month_runtime=Sum(runtime, filter=filter_this_month),
            total_cpu_usage=Sum("taskreport__cpu_cost"),
            this_week_cpu_usage=Sum("taskreport__cpu_cost", filter=filter_this_week),
            this_month_cpu_usage=Sum("taskreport__cpu_cost", filter=filter_this_month),
            total_gpu_usage=Sum("taskreport__gpu_cost"),
            this_week_gpu_usage=Sum("taskreport__gpu_cost", filter=filter_this_week),
            this_month_gpu_usage=Sum("taskreport__gpu_cost", filter=filter_this_month),
            vm_segment_total=Count("taskreport", filter=filter_vm_segment),
            vm_segment_this_week=Count(
                "taskreport", filter=filter_vm_segment & filter_this_week
            ),
            vm_segment_this_month=Count(
                "taskreport", filter=filter_vm_segment & filter_this_month
            ),
            vm_transcribe_total=Count("taskreport", filter=filter_vm_transcribe),
            vm_transcribe_this_week=Count(
                "taskreport", filter=filter_vm_transcribe & filter_this_week
            ),
            vm_transcribe_this_month=Count(
                "taskreport", filter=filter_vm_transcribe & filter_this_month
            ),
            vm_segtrain_total=Count("taskreport", filter=filter_vm_segtrain),
            vm_segtrain_this_week=Count(
                "taskreport", filter=filter_vm_segtrain & filter_this_week
            ),
            vm_segtrain_this_month=Count(
                "taskreport", filter=filter_vm_segtrain & filter_this_month
            ),
            vm_train_total=Count("taskreport", filter=filter_vm_train),
            vm_train_this_week=Count(
                "taskreport", filter=filter_vm_train & filter_this_week
            ),
            vm_train_this_month=Count(
                "taskreport", filter=filter_vm_train & filter_this_month
            ),
            hpc_segtrain_total=Count("taskreport", filter=filter_hpc_segtrain),
            hpc_segtrain_this_week=Count(
                "taskreport", filter=filter_hpc_segtrain & filter_this_week
            ),
            hpc_segtrain_this_month=Count(
                "taskreport", filter=filter_hpc_segtrain & filter_this_month
            ),
            hpc_train_total=Count("taskreport", filter=filter_hpc_train),
            hpc_train_this_week=Count(
                "taskreport", filter=filter_hpc_train & filter_this_week
            ),
            hpc_train_this_month=Count(
                "taskreport", filter=filter_hpc_train & filter_this_month
            ),
        )
        .order_by("username")
    )


def user_to_csv_row(user):
    """Convert an annotated user object to a CSV row dict."""
    return {
        "username": user.username,
        "disk_usage_bytes": user.disk_usage or 0,
        "total_tasks": user.total_tasks,
        "this_week_tasks": user.this_week_tasks,
        "this_month_tasks": user.this_month_tasks,
        "total_runtime_seconds": int(user.total_runtime.total_seconds())
        if user.total_runtime
        else 0,
        "this_week_runtime_seconds": int(user.this_week_runtime.total_seconds())
        if user.this_week_runtime
        else 0,
        "this_month_runtime_seconds": int(user.this_month_runtime.total_seconds())
        if user.this_month_runtime
        else 0,
        "total_cpu_usage": user.total_cpu_usage or 0,
        "this_week_cpu_usage": user.this_week_cpu_usage or 0,
        "this_month_cpu_usage": user.this_month_cpu_usage or 0,
        "total_gpu_usage": user.total_gpu_usage or 0,
        "this_week_gpu_usage": user.this_week_gpu_usage or 0,
        "this_month_gpu_usage": user.this_month_gpu_usage or 0,
        "vm_segment_total": user.vm_segment_total,
        "vm_segment_this_week": user.vm_segment_this_week,
        "vm_segment_this_month": user.vm_segment_this_month,
        "vm_transcribe_total": user.vm_transcribe_total,
        "vm_transcribe_this_week": user.vm_transcribe_this_week,
        "vm_transcribe_this_month": user.vm_transcribe_this_month,
        "vm_segtrain_total": user.vm_segtrain_total,
        "vm_segtrain_this_week": user.vm_segtrain_this_week,
        "vm_segtrain_this_month": user.vm_segtrain_this_month,
        "vm_train_total": user.vm_train_total,
        "vm_train_this_week": user.vm_train_this_week,
        "vm_train_this_month": user.vm_train_this_month,
        "hpc_segtrain_total": user.hpc_segtrain_total,
        "hpc_segtrain_this_week": user.hpc_segtrain_this_week,
        "hpc_segtrain_this_month": user.hpc_segtrain_this_month,
        "hpc_train_total": user.hpc_train_total,
        "hpc_train_this_week": user.hpc_train_this_week,
        "hpc_train_this_month": user.hpc_train_this_month,
    }
