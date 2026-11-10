from django.conf import settings
from django.db import models


class TaskReport(models.Model):
    """Subset of eScriptorium's reporting.TaskReport fields."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    task_id = models.CharField(max_length=64, blank=True, null=True)
    method = models.CharField(max_length=512, blank=True, null=True)
    started_at = models.DateTimeField(null=True)
    done_at = models.DateTimeField(null=True)
    cpu_cost = models.FloatField(blank=True, null=True)
