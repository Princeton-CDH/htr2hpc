from django.conf import settings
from django.db import models


class OcrModel(models.Model):
    """Subset of eScriptorium's core.OcrModel fields."""

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL
    )
    file_size = models.BigIntegerField()
