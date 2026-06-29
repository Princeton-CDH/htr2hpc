import datetime
import logging
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

EXPORT_FILE_RETENTION_DEFAULT = 30

logger = logging.getLogger(__name__)


def delete_old_exports(media_root, retention_days, dry_run=False):
    """Delete export files under media_root/users/ older than retention_days.

    Returns a (count, total_bytes) tuple of files deleted (or that would be
    deleted when dry_run=True).
    """
    cutoff = datetime.datetime.now() - datetime.timedelta(days=retention_days)
    users_dir = Path(media_root) / "users"

    if not users_dir.is_dir():
        return 0, 0

    count = 0
    total_bytes = 0

    for user_dir in users_dir.iterdir():
        if not user_dir.is_dir():
            continue
        for entry in user_dir.iterdir():
            if not entry.name.startswith("export_"):
                continue
            stat = entry.stat()
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
            if mtime < cutoff:
                size = stat.st_size
                logger.info("Deleting %s (%d bytes)", entry, size)
                if not dry_run:
                    entry.unlink()
                count += 1
                total_bytes += size

    return count, total_bytes


class Command(BaseCommand):
    help = (
        f"Delete export files older than settings.EXPORT_FILE_RETENTION days "
        f"(default: {EXPORT_FILE_RETENTION_DEFAULT} days)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Report what would be deleted without actually deleting.",
        )

    def handle(self, *args, **kwargs):
        dry_run = kwargs["dry_run"]
        retention = getattr(
            settings, "EXPORT_FILE_RETENTION", EXPORT_FILE_RETENTION_DEFAULT
        )

        if retention == 0:
            self.stdout.write(
                "EXPORT_FILE_RETENTION set to 0. Nothing will be cleaned up."
            )
            return

        if not (Path(settings.MEDIA_ROOT) / "users").is_dir():
            self.stdout.write("No users media directory found; nothing to clean up.")
            return

        count, total_bytes = delete_old_exports(settings.MEDIA_ROOT, retention, dry_run)
        action = "Would delete" if dry_run else "Deleted"
        self.stdout.write(
            f"{action} {count} export file(s), freeing {total_bytes} bytes."
        )
