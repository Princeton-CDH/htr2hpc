from django import forms
from django.conf import settings

from htr2hpc.exports import REPORT_TYPE_CHOICES


class UsageReportForm(forms.Form):
    """Options for downloading the staff usage report CSV."""

    start = forms.DateField(
        required=False,
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
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["start"].initial = settings.USAGE_REPORT_DEFAULT_START

    def clean_start(self):
        """Fall back to the configured default when no start date is given."""
        return self.cleaned_data["start"] or settings.USAGE_REPORT_DEFAULT_START
