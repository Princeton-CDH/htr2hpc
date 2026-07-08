import datetime
from pathlib import Path
from typing import Generator

from django.conf import settings
from django.core.management.base import BaseCommand


def get_old_exports(
    users_dir: Path, cutoff: datetime.datetime
) -> Generator[tuple[Path, int], None, None]:
    """Yield (path, size_in_bytes) for export files older than cutoff."""
    for entry in users_dir.glob("*/export_*.zip"):
        stat = entry.stat()
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        if mtime < cutoff:
            yield entry, stat.st_size


class Command(BaseCommand):
    help = "Delete export files older than settings.EXPORT_FILE_RETENTION hours."
    v_normal = 1

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Report what would be deleted without actually deleting.",
        )

    def delete_old_exports(
        self, export_dir: Path, retention_hours: int
    ) -> tuple[int, int]:
        """Delete export files in export_dir older than retention_hours.

        Returns a (count, total_bytes) tuple of files deleted (or that would be
        deleted when dry_run=True).
        """
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=retention_hours)

        if not export_dir.is_dir():
            return 0, 0

        count = 0
        total_bytes = 0

        for entry, size in get_old_exports(export_dir, cutoff):
            if self.verbosity >= self.v_normal + 1:
                action = "Would delete" if self.dry_run else "Deleting"
                self.stdout.write(f"{action} {entry} ({size} bytes)")
            if not self.dry_run:
                try:
                    entry.unlink()
                except OSError:
                    continue
            count += 1
            total_bytes += size

        return count, total_bytes

    def handle(self, *args, **kwargs):
        self.dry_run = kwargs["dry_run"]
        self.verbosity = kwargs["verbosity"]
        retention = settings.EXPORT_FILE_RETENTION

        if retention == 0:
            self.stdout.write(
                "EXPORT_FILE_RETENTION set to 0. Nothing will be cleaned up."
            )
            return

        export_dir = Path(settings.MEDIA_ROOT) / "users"
        if not export_dir.is_dir():
            self.stdout.write("No users media directory found; nothing to clean up.")
            return

        count, total_bytes = self.delete_old_exports(export_dir, retention)

        if self.verbosity >= self.v_normal:
            action = "Would delete" if self.dry_run else "Deleted"
            self.stdout.write(
                f"{action} {count} export file(s), freeing {total_bytes} bytes."
            )
