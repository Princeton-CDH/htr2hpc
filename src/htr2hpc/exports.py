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
    "last_week_tasks",
    "last_month_tasks",
    "total_runtime_seconds",
    "last_week_runtime_seconds",
    "last_month_runtime_seconds",
    "total_cpu_usage",
    "last_week_cpu_usage",
    "last_month_cpu_usage",
    "total_gpu_usage",
    "last_week_gpu_usage",
    "last_month_gpu_usage",
    "vm_segment_total",
    "vm_segment_last_week",
    "vm_segment_last_month",
    "vm_transcribe_total",
    "vm_transcribe_last_week",
    "vm_transcribe_last_month",
    "vm_segtrain_total",
    "vm_segtrain_last_week",
    "vm_segtrain_last_month",
    "vm_train_total",
    "vm_train_last_week",
    "vm_train_last_month",
    "hpc_segtrain_total",
    "hpc_segtrain_last_week",
    "hpc_segtrain_last_month",
    "hpc_train_total",
    "hpc_train_last_week",
    "hpc_train_last_month",
]


def get_usage_data():  # pragma: no cover
    """Return annotated user queryset with VM and HPC usage stats.

    Uses a Subquery for disk_usage to avoid row multiplication from
    joining both ocrmodel and taskreport relations simultaneously.
    """
    now = datetime.now(tz=UTC)
    filter_last_week = Q(taskreport__started_at__gte=now - timedelta(days=7))
    filter_last_month = Q(taskreport__started_at__gte=now - timedelta(days=30))
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
            last_week_tasks=Count("taskreport", filter=filter_last_week),
            last_month_tasks=Count("taskreport", filter=filter_last_month),
            total_runtime=Sum(runtime),
            last_week_runtime=Sum(runtime, filter=filter_last_week),
            last_month_runtime=Sum(runtime, filter=filter_last_month),
            total_cpu_usage=Sum("taskreport__cpu_cost"),
            last_week_cpu_usage=Sum("taskreport__cpu_cost", filter=filter_last_week),
            last_month_cpu_usage=Sum("taskreport__cpu_cost", filter=filter_last_month),
            total_gpu_usage=Sum("taskreport__gpu_cost"),
            last_week_gpu_usage=Sum("taskreport__gpu_cost", filter=filter_last_week),
            last_month_gpu_usage=Sum("taskreport__gpu_cost", filter=filter_last_month),
            vm_segment_total=Count("taskreport", filter=filter_vm_segment),
            vm_segment_last_week=Count(
                "taskreport", filter=filter_vm_segment & filter_last_week
            ),
            vm_segment_last_month=Count(
                "taskreport", filter=filter_vm_segment & filter_last_month
            ),
            vm_transcribe_total=Count("taskreport", filter=filter_vm_transcribe),
            vm_transcribe_last_week=Count(
                "taskreport", filter=filter_vm_transcribe & filter_last_week
            ),
            vm_transcribe_last_month=Count(
                "taskreport", filter=filter_vm_transcribe & filter_last_month
            ),
            vm_segtrain_total=Count("taskreport", filter=filter_vm_segtrain),
            vm_segtrain_last_week=Count(
                "taskreport", filter=filter_vm_segtrain & filter_last_week
            ),
            vm_segtrain_last_month=Count(
                "taskreport", filter=filter_vm_segtrain & filter_last_month
            ),
            vm_train_total=Count("taskreport", filter=filter_vm_train),
            vm_train_last_week=Count(
                "taskreport", filter=filter_vm_train & filter_last_week
            ),
            vm_train_last_month=Count(
                "taskreport", filter=filter_vm_train & filter_last_month
            ),
            hpc_segtrain_total=Count("taskreport", filter=filter_hpc_segtrain),
            hpc_segtrain_last_week=Count(
                "taskreport", filter=filter_hpc_segtrain & filter_last_week
            ),
            hpc_segtrain_last_month=Count(
                "taskreport", filter=filter_hpc_segtrain & filter_last_month
            ),
            hpc_train_total=Count("taskreport", filter=filter_hpc_train),
            hpc_train_last_week=Count(
                "taskreport", filter=filter_hpc_train & filter_last_week
            ),
            hpc_train_last_month=Count(
                "taskreport", filter=filter_hpc_train & filter_last_month
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
        "last_week_tasks": user.last_week_tasks,
        "last_month_tasks": user.last_month_tasks,
        "total_runtime_seconds": int(user.total_runtime.total_seconds())
        if user.total_runtime
        else 0,
        "last_week_runtime_seconds": int(user.last_week_runtime.total_seconds())
        if user.last_week_runtime
        else 0,
        "last_month_runtime_seconds": int(user.last_month_runtime.total_seconds())
        if user.last_month_runtime
        else 0,
        "total_cpu_usage": user.total_cpu_usage or 0,
        "last_week_cpu_usage": user.last_week_cpu_usage or 0,
        "last_month_cpu_usage": user.last_month_cpu_usage or 0,
        "total_gpu_usage": user.total_gpu_usage or 0,
        "last_week_gpu_usage": user.last_week_gpu_usage or 0,
        "last_month_gpu_usage": user.last_month_gpu_usage or 0,
        "vm_segment_total": user.vm_segment_total,
        "vm_segment_last_week": user.vm_segment_last_week,
        "vm_segment_last_month": user.vm_segment_last_month,
        "vm_transcribe_total": user.vm_transcribe_total,
        "vm_transcribe_last_week": user.vm_transcribe_last_week,
        "vm_transcribe_last_month": user.vm_transcribe_last_month,
        "vm_segtrain_total": user.vm_segtrain_total,
        "vm_segtrain_last_week": user.vm_segtrain_last_week,
        "vm_segtrain_last_month": user.vm_segtrain_last_month,
        "vm_train_total": user.vm_train_total,
        "vm_train_last_week": user.vm_train_last_week,
        "vm_train_last_month": user.vm_train_last_month,
        "hpc_segtrain_total": user.hpc_segtrain_total,
        "hpc_segtrain_last_week": user.hpc_segtrain_last_week,
        "hpc_segtrain_last_month": user.hpc_segtrain_last_month,
        "hpc_train_total": user.hpc_train_total,
        "hpc_train_last_week": user.hpc_train_last_week,
        "hpc_train_last_month": user.hpc_train_last_month,
    }
