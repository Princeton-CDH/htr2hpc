"""Management command to export per-job HPC resource stats from sacct."""

import sys
from datetime import date
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from htr2hpc.exports import hpc_task_usernames
from htr2hpc.train.jobstats import get_job_stats, job_stats_to_csv


class Command(BaseCommand):
    help = (
        "Export per-job HPC resource stats for a date range. Runs sacct, so it "
        "must be run where sacct is available. Defaults to users who ran HPC "
        "training tasks in the period."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--start", required=True, metavar="YYYY-MM-DD", help="Start date."
        )
        parser.add_argument(
            "--end", metavar="YYYY-MM-DD", help="End date. Defaults to today."
        )
        parser.add_argument(
            "--users",
            metavar="NETID[,NETID,...]",
            help="Comma-separated NetIDs to report on, instead of users who ran "
            "HPC training tasks in the period.",
        )
        parser.add_argument(
            "--output", metavar="FILE", help="File to write CSV to. Defaults to stdout."
        )

    def handle(self, *args, **options):
        try:
            start = date.fromisoformat(options["start"])
            end = (
                date.fromisoformat(options["end"])
                if options["end"]
                else timezone.localdate()
            )
        except ValueError as err:
            raise CommandError(f"Invalid date: {err}") from err

        if options["users"]:
            usernames = [u.strip() for u in options["users"].split(",") if u.strip()]
        else:
            usernames = hpc_task_usernames(start, end)
        if not usernames:
            raise CommandError("No users to report on.")

        try:
            rows = get_job_stats(start.isoformat(), end.isoformat(), usernames)
        except Exception as err:
            raise CommandError(f"sacct query failed: {err}") from err
        self.stderr.write(f"Found {len(rows)} job(s) for {len(usernames)} user(s).")

        if options["output"]:
            with Path(options["output"]).open("w", newline="") as outfile:
                job_stats_to_csv(rows, outfile)
        else:
            job_stats_to_csv(rows, sys.stdout)
