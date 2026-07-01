import datetime
from pathlib import Path
from typing import Generator

from django.conf import settings
from django.core.management.base import BaseCommand

EXPORT_FILE_RETENTION_DEFAULT = 168  # 1 week in hours

VERBOSITY_QUIET = 0
VERBOSITY_NORMAL = 1
VERBOSITY_VERBOSE = 2


def get_old_exports(
    users_dir: Path, cutoff: datetime.datetime
) -> Generator[tuple[Path, int], None, None]:
    """Yield (path, size_in_bytes) for export files older than cutoff."""
    for entry in users_dir.glob("*/export_*.zip"):
        stat = entry.stat()
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        if mtime < cutoff:
            yield entry, stat.st_size


def delete_old_exports(
    media_root: Path | str, retention_hours: int, dry_run: bool = False
) -> tuple[int, int]:
    """Delete export files under media_root/users/ older than retention_hours.

    Returns a (count, total_bytes) tuple of files deleted (or that would be
    deleted when dry_run=True).
    """
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=retention_hours)
    users_dir = Path(media_root) / "users"

    if not users_dir.is_dir():
        return 0, 0

    count = 0
    total_bytes = 0

    for entry, size in get_old_exports(users_dir, cutoff):
        if not dry_run:
            entry.unlink()
        count += 1
        total_bytes += size

    return count, total_bytes


class Command(BaseCommand):
    help = "Delete export files older than settings.EXPORT_FILE_RETENTION hours."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Report what would be deleted without actually deleting.",
        )

    def handle(self, *args, **kwargs):
        dry_run = kwargs["dry_run"]
        verbosity = kwargs["verbosity"]
        retention = settings.EXPORT_FILE_RETENTION

        if retention == 0:
            self.stdout.write(
                "EXPORT_FILE_RETENTION set to 0. Nothing will be cleaned up."
            )
            return

        users_dir = Path(settings.MEDIA_ROOT) / "users"
        if not users_dir.is_dir():
            self.stdout.write("No users media directory found; nothing to clean up.")
            return

        cutoff = datetime.datetime.now() - datetime.timedelta(hours=retention)
        count = 0
        total_bytes = 0

        for entry, size in get_old_exports(users_dir, cutoff):
            if verbosity >= VERBOSITY_VERBOSE:
                action = "Would delete" if dry_run else "Deleting"
                self.stdout.write(f"{action} {entry} ({size} bytes)")
            if not dry_run:
                entry.unlink()
            count += 1
            total_bytes += size

        if verbosity >= VERBOSITY_NORMAL:
            action = "Would delete" if dry_run else "Deleted"
            self.stdout.write(
                f"{action} {count} export file(s), freeing {total_bytes} bytes."
            )
