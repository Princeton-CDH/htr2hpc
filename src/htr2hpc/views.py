import csv
from itertools import chain

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import (
    HttpResponseBadRequest,
    HttpResponseRedirect,
    StreamingHttpResponse,
)
from django.urls import reverse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods

from htr2hpc.exports import (
    REPORT_FILENAME_LABELS,
    USAGE_REPORT_FIELDS,
    get_usage_report,
    report_row,
)
from htr2hpc.forms import UsageReportForm
from htr2hpc.tasks import hpc_user_setup


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
