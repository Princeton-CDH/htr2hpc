from django import forms
from django.conf import settings

from htr2hpc.exports import REPORT_TYPE_CHOICES


class UsageReportForm(forms.Form):
    """Options for downloading the staff usage report CSV."""

    start = forms.DateField(
        widget=forms.DateInput(
            attrs={"type": "date", "class": "form-control form-control-sm"},
            format="%Y-%m-%d",
        ),
    )
    report_type = forms.ChoiceField(
        choices=REPORT_TYPE_CHOICES,
        widget=forms.Select(
            attrs={"class": "form-control form-control-sm", "aria-label": "Report type"}
        ),
        help_text=(
            "Total covers everything from the start date. Monthly and weekly "
            "only include completed months and weeks (weeks start on Monday), "
            "so a partial first period and the current period are left out."
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pre-fill the start date when a default is configured
        default_start = getattr(settings, "USAGE_REPORT_DEFAULT_START", None)
        if default_start:
            self.fields["start"].initial = default_start


class HPCJobStatsForm(forms.Form):
    """Date range for downloading the staff HPC job stats CSV."""

    start = forms.DateField(
        widget=forms.DateInput(
            attrs={"type": "date", "class": "form-control form-control-sm"},
            format="%Y-%m-%d",
        ),
    )
    end = forms.DateField(
        required=False,
        widget=forms.DateInput(
            attrs={"type": "date", "class": "form-control form-control-sm"},
            format="%Y-%m-%d",
        ),
        help_text="Defaults to today.",
    )

    def clean(self):
        cleaned = super().clean()
        start, end = cleaned.get("start"), cleaned.get("end")
        if start and end and end < start:
            self.add_error("end", "End date must not be before the start date.")
        return cleaned
