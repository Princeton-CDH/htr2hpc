import csv
from datetime import UTC, datetime

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect, StreamingHttpResponse
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods

from htr2hpc.exports import USAGE_REPORT_FIELDS, get_usage_data, user_to_csv_row
from htr2hpc.tasks import hpc_user_setup


class _Echo:
    """A file-like object that returns whatever is written to it.

    Used with csv.writer to yield individual CSV rows as strings
    without buffering, per the Django streaming CSV docs.
    """

    def write(self, value):
        return value


def _csv_rows(queryset):
    """Yield CSV rows as strings, starting with headers."""
    writer = csv.writer(_Echo())
    yield writer.writerow(USAGE_REPORT_FIELDS)
    for user in queryset:
        yield writer.writerow(list(user_to_csv_row(user).values()))


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
class UsageReportView(View):
    """Admin-only CSV export of per-user VM and HPC task usage stats.

    Restricted to staff via staff_member_required decorator.
    The download link is placed on the existing Leaderboard page
    via a template override in templates/reporting/quotas_leaderboard.html.
    """

    def get(self, request, *args, **kwargs):  # pragma: no cover
        today = datetime.now(tz=UTC).date()
        response = StreamingHttpResponse(
            _csv_rows(get_usage_data()),
            content_type="text/csv",
        )
        response["Content-Disposition"] = (
            f'attachment; filename="htr-usage-{today}.csv"'
        )
        return response
