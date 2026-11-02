"""Management command to export per-job HPC resource stats from sacct."""

import sys
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from htr2hpc.train.jobstats import (
    job_stats_to_csv,
    sacct_to_job_stats,
)

User = get_user_model()


class Command(BaseCommand):
    help = (
        "Export per-job HPC resource stats for a date range. "
        "Queries sacct and decodes Jobstats data from the admincomment field. "
        "Output is written to stdout or --output file."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--start",
            required=True,
            metavar="YYYY-MM-DD",
            help="Start date (inclusive) for sacct query.",
        )
        parser.add_argument(
            "--end",
            required=True,
            metavar="YYYY-MM-DD",
            help="End date (inclusive) for sacct query.",
        )
        parser.add_argument(
            "--account",
            metavar="SLURM_ACCOUNT",
            help="Slurm account to filter by (e.g. pustaff). "
            "At least one of --account or --users is required.",
        )
        parser.add_argument(
            "--users",
            metavar="NETID[,NETID,...]",
            help="Comma-separated list of NetIDs to filter by. "
            "At least one of --account or --users is required.",
        )
        parser.add_argument(
            "--active-users-only",
            action="store_true",
            default=False,
            help="Restrict --account query to NetIDs of currently active htr2hpc users.",
        )
        parser.add_argument(
            "--output",
            metavar="FILE",
            help="Path to write CSV output. Defaults to stdout.",
        )

    def handle(self, *args, **kwargs):
        start = kwargs["start"]
        end = kwargs["end"]
        account = kwargs["account"]
        users_arg = kwargs["users"]
        active_users_only = kwargs["active_users_only"]
        output_path = kwargs["output"]

        if not account and not users_arg:
            raise CommandError("Provide at least one of --account or --users.")

        usernames: list[str] | None = None
        if users_arg:
            usernames = [u.strip() for u in users_arg.split(",") if u.strip()]

        if active_users_only:
            if not account:
                raise CommandError("--active-users-only requires --account.")
            active_netids = list(
                User.objects.filter(is_active=True).values_list("username", flat=True)
            )
            if not active_netids:
                self.stderr.write("No active users found; nothing to query.")
                return
            usernames = active_netids
            self.stderr.write(f"Filtering to {len(active_netids)} active user(s).")

        self.stderr.write(
            f"Querying sacct: start={start} end={end}"
            + (f" account={account}" if account else "")
            + (f" users={','.join(usernames)}" if usernames else "")
        )

        try:
            rows = sacct_to_job_stats(start, end, usernames=usernames, account=account)
        except Exception as exc:
            raise CommandError(f"sacct query failed: {exc}") from exc

        self.stderr.write(f"Found {len(rows)} job(s).")

        if output_path:
            with Path(output_path).open("w", newline="") as f:
                job_stats_to_csv(rows, f)
            self.stderr.write(f"Written to {output_path}")
        else:
            job_stats_to_csv(rows, sys.stdout)
