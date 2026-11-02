from datetime import UTC, datetime

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import (
    HttpResponseBadRequest,
    HttpResponseRedirect,
    StreamingHttpResponse,
)
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods

from htr2hpc.tasks import hpc_user_setup
from htr2hpc.train.jobstats import job_stats_to_csv_string, sacct_to_job_stats


class _Echo:
    """A file-like object that returns whatever is written to it.

    Used with csv.writer to yield individual CSV rows as strings
    without buffering, per the Django streaming CSV docs.
    """

    def write(self, value):
        return value


@login_required
@require_http_methods(["POST"])
def remote_user_setup(request):  # pragma: no cover
    # it seems that the taskreport must be associated with a document,
    # so skip the task reporting logic here and just use notifications

    # queue the celery setup task
    hpc_user_setup.delay(user_pk=request.user.pk)
    # redirect back to the profile page
    redirect = HttpResponseRedirect(reverse("profile-api-key"))
    redirect.status_code = 303  # See other
    return redirect


@method_decorator(staff_member_required, name="dispatch")
class HpcJobStatsView(View):
    """Admin-only CSV export of per-job HPC resource stats from sacct.

    Query parameters:
      - start:   YYYY-MM-DD start date (required)
      - end:     YYYY-MM-DD end date   (required)
      - account: Slurm account (e.g. pustaff)
      - users:   comma-separated NetIDs

    At least one of ``account`` or ``users`` is required.
    """

    def get(self, request, *args, **kwargs):  # pragma: no cover
        start = request.GET.get("start", "")
        end = request.GET.get("end", "")
        account = request.GET.get("account", "") or None
        users_param = request.GET.get("users", "")
        usernames = [u.strip() for u in users_param.split(",") if u.strip()] or None

        if not start or not end:
            return HttpResponseBadRequest("start and end query parameters are required")
        if not account and not usernames:
            return HttpResponseBadRequest(
                "account or users query parameter is required"
            )

        rows = sacct_to_job_stats(start, end, usernames=usernames, account=account)
        csv_content = job_stats_to_csv_string(rows)

        today = datetime.now(tz=UTC).date()
        response = StreamingHttpResponse(
            (csv_content,),
            content_type="text/csv",
        )
        response["Content-Disposition"] = (
            f'attachment; filename="hpc-job-stats-{today}.csv"'
        )
        return response
