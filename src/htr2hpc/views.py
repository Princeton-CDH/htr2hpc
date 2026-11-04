import csv
import io
import zipfile
from datetime import UTC, datetime

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods

from htr2hpc.exports import (
    MONTHLY_FIELDS,
    TOTALS_FIELDS,
    get_completed_months,
    get_monthly_data,
    get_totals_data,
    monthly_row,
    totals_row,
)
from htr2hpc.tasks import hpc_user_setup


def _write_csv(fields, rows) -> bytes:
    """Write a list of row dicts to CSV and return as bytes."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode()


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
    """Admin-only ZIP export of per-user VM and HPC task usage stats.

    The ZIP contains:
    - totals.csv: all-time stats per user (from USAGE_REPORT_START)
    - YYYY-MM.csv: one file per completed calendar month

    Restricted to staff via staff_member_required decorator.
    The download link is placed on the existing Leaderboard page
    via a template override in templates/reporting/quotas_leaderboard.html.
    """

    def get(self, request, *args, **kwargs):  # pragma: no cover
        today = datetime.now(tz=UTC).date()
        zip_buf = io.BytesIO()

        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # totals CSV
            totals_csv = _write_csv(
                TOTALS_FIELDS,
                [totals_row(u) for u in get_totals_data()],
            )
            zf.writestr("totals.csv", totals_csv)

            # one CSV per completed calendar month
            for year, month in get_completed_months():
                monthly_csv = _write_csv(
                    MONTHLY_FIELDS,
                    [monthly_row(u) for u in get_monthly_data(year, month)],
                )
                zf.writestr(f"{year:04d}-{month:02d}.csv", monthly_csv)

        zip_buf.seek(0)
        response = HttpResponse(
            zip_buf.read(),
            content_type="application/zip",
        )
        response["Content-Disposition"] = (
            f'attachment; filename="htr-usage-{today}.zip"'
        )
        return response
