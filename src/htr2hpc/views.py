import csv
import logging
from itertools import chain

from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import (
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseRedirect,
    StreamingHttpResponse,
)
from django.urls import reverse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods
from fabric import Connection

from htr2hpc.exports import (
    REPORT_FILENAME_LABELS,
    USAGE_REPORT_FIELDS,
    get_usage_report,
    hpc_task_usernames,
    report_row,
)
from htr2hpc.forms import HPCJobStatsForm, UsageReportForm
from htr2hpc.tasks import hpc_user_setup
from htr2hpc.train.jobstats import (
    JOB_STATS_FIELDS,
    get_job_stats,
    job_stats_csv_rows,
)

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["POST"])
def remote_user_setup(request):
    # it seems that the taskreport must be associated with a document,
    # so skip the task reporting logic here and just use notifications

    # queue the celery setup task
    hpc_user_setup.delay(user_pk=request.user.pk)
    # redirect back to the profile page
    redirect = HttpResponseRedirect(reverse("profile-api-key"))
    redirect.status_code = 303  # See other
    return redirect


class Echo:
    """File-like object that returns the written value instead of buffering it,
    so csv.writer output can be streamed row by row."""

    def write(self, value):
        return value


@method_decorator(staff_member_required, name="dispatch")
class UsageReportView(View):
    """Staff-only CSV download of per-user VM and HPC usage stats.

    Takes GET parameters from :class:`~htr2hpc.forms.UsageReportForm`:
    ``report_type`` and ``start``, both required. Returns a 400 if they are
    missing or invalid. The download form is shown on the Leaderboard page.
    """

    def get(self, request, *args, **kwargs):
        form = UsageReportForm(request.GET)
        if not form.is_valid():
            return HttpResponseBadRequest(form.errors.as_text())

        start = form.cleaned_data["start"]
        report_type = form.cleaned_data["report_type"]
        rows = get_usage_report(start, report_type)

        writer = csv.DictWriter(Echo(), fieldnames=USAGE_REPORT_FIELDS)
        header = dict(zip(USAGE_REPORT_FIELDS, USAGE_REPORT_FIELDS, strict=True))
        csv_rows = chain(
            [header], (report_row(r, report_type) for r in rows.iterator())
        )
        lines = (writer.writerow(row) for row in csv_rows)

        filename = (
            f"htr-usage-{REPORT_FILENAME_LABELS[report_type]}"
            f"-{start.isoformat()}-{timezone.localdate().isoformat()}.csv"
        )
        return StreamingHttpResponse(
            lines,
            content_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


@method_decorator(staff_member_required, name="dispatch")
class HPCJobStatsView(View):
    """Staff-only CSV download of per-job HPC resource stats.

    POST only, since it runs ``sacct`` on the HPC over SSH. Takes ``start``
    (required) and ``end`` (optional, defaults to today) dates from
    :class:`~htr2hpc.forms.HPCJobStatsForm`. Reports Slurm jobs for users who
    ran HPC training tasks in that period.
    """

    http_method_names = ("post",)

    def post(self, request, *args, **kwargs):
        form = HPCJobStatsForm(request.POST)
        if not form.is_valid():
            return HttpResponseBadRequest(form.errors.as_text())

        start = form.cleaned_data["start"]
        end = form.cleaned_data["end"] or timezone.localdate()
        usernames = hpc_task_usernames(start, end)

        rows = []
        if usernames:
            try:
                # connect as the staff member, as the training tasks do for users
                with Connection(
                    host=settings.HPC_HOSTNAME,
                    user=request.user.username,
                    connect_timeout=10,
                    connect_kwargs={"key_filename": settings.HPC_SSH_KEYFILE},
                ) as conn:
                    rows = get_job_stats(
                        start.isoformat(), end.isoformat(), usernames, connection=conn
                    )
            except Exception:
                logger.exception("Failed to get HPC job stats")
                return HttpResponse(
                    "Could not get job stats from the HPC; see the logs for details.",
                    status=502,
                    content_type="text/plain",
                )

        writer = csv.DictWriter(Echo(), fieldnames=JOB_STATS_FIELDS)
        lines = (writer.writerow(row) for row in job_stats_csv_rows(rows))
        filename = f"htr-hpc-jobs-{start.isoformat()}-{end.isoformat()}.csv"
        return StreamingHttpResponse(
            lines,
            content_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
