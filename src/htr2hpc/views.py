import csv
from datetime import UTC, datetime

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from django.views.generic import TemplateView

from htr2hpc.exports import USAGE_REPORT_FIELDS, get_usage_data, user_to_csv_row
from htr2hpc.tasks import hpc_user_setup


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


class UsageReportView(TemplateView):
    """Admin-only page to download a CSV of per-user VM and HPC task usage stats.

    Restricted to staff via staff_member_required in urls.py.
    GET returns the CSV directly.
    """

    template_name = "htr2hpc/usage_report.html"

    def get(self, request, *args, **kwargs):  # pragma: no cover
        today = datetime.now(tz=UTC).date()
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = (
            f'attachment; filename="htr-usage-{today}.csv"'
        )
        writer = csv.DictWriter(response, fieldnames=USAGE_REPORT_FIELDS)
        writer.writeheader()
        for user in get_usage_data():
            writer.writerow(user_to_csv_row(user))
        return response
