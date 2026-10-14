import csv
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db.models import Avg, Count, Max
from django.utils import timezone

from htr2hpc.models import HpcJobStats


class Command(BaseCommand):
    help = "Report HPC resource usage per user (job counts, memory, storage)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--user",
            metavar="USERNAME",
            help="Filter report to a specific user.",
        )
        parser.add_argument(
            "--days",
            type=int,
            default=None,
            metavar="N",
            help="Only include jobs from the last N days. Defaults to all time.",
        )
        parser.add_argument(
            "--format",
            choices=["text", "csv"],
            default="text",
            help="Output format: 'text' (default) or 'csv'.",
        )

    def handle(self, *args, **options):
        qs = HpcJobStats.objects.select_related("user")

        if options["user"]:
            User = get_user_model()
            try:
                user = User.objects.get(username=options["user"])
            except User.DoesNotExist:
                self.stderr.write(f"User '{options['user']}' not found.")
                return
            qs = qs.filter(user=user)

        if options["days"]:
            since = timezone.now() - timedelta(days=options["days"])
            qs = qs.filter(created_at__gte=since)

        # aggregate per user + job type
        rows = (
            qs.values("user__username", "job_type")
            .annotate(
                job_count=Count("pk"),
                avg_memory_mb=Avg("max_memory_mb"),
                max_memory_mb=Max("max_memory_mb"),
                max_storage_mb=Max("scratch_storage_mb"),
            )
            .order_by("user__username", "job_type")
        )

        rows = list(rows)
        if not rows:
            self.stdout.write("No HPC job stats found.")
            return

        if options["format"] == "csv":
            self._write_csv(rows)
        else:
            self._write_text(rows)

    def _write_text(self, rows):
        header = f"{'User':<20} {'Job type':<12} {'Jobs':>6} {'Avg mem (MB)':>14} {'Max mem (MB)':>14} {'Storage (MB)':>14}"
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for row in rows:
            self.stdout.write(
                f"{row['user__username']:<20} "
                f"{row['job_type']:<12} "
                f"{row['job_count']:>6} "
                f"{self._fmt(row['avg_memory_mb']):>14} "
                f"{self._fmt(row['max_memory_mb']):>14} "
                f"{self._fmt(row['max_storage_mb']):>14}"
            )

    def _write_csv(self, rows):
        writer = csv.DictWriter(
            self.stdout,
            fieldnames=[
                "username",
                "job_type",
                "job_count",
                "avg_memory_mb",
                "max_memory_mb",
                "max_storage_mb",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "username": row["user__username"],
                    "job_type": row["job_type"],
                    "job_count": row["job_count"],
                    "avg_memory_mb": row["avg_memory_mb"] or "",
                    "max_memory_mb": row["max_memory_mb"] or "",
                    "max_storage_mb": row["max_storage_mb"] or "",
                }
            )

    def _fmt(self, value):
        return f"{value:.1f}" if value is not None else "N/A"
