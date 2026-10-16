from typing import ClassVar

from django.conf import settings
from django.db import models


class SiteContent(models.Model):
    """Admin-editable content snippets injected into specific pages.

    Allows staff to update page content (contact info, announcements, model
    descriptions) without a code change or deployment. Each record is identified
    by a slug key used in the ``{% site_content %}`` template tag.
    """

    key = models.SlugField(max_length=100, unique=True)
    label = models.CharField(max_length=200)
    body = models.TextField(
        blank=True,
        help_text="HTML content to inject into the page. Leave blank to hide the slot.",
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "site content"
        verbose_name_plural = "site content"
        ordering: ClassVar[list] = ["key"]

    def __str__(self):
        return self.label or self.key


class HpcJobStats(models.Model):
    """Stores resource usage statistics for completed HPC training jobs.

    Populated by the celery worker after each training job completes.
    Tracks memory usage, elapsed time, GPU usage, and scratch storage per user.
    """

    task_report_id = models.IntegerField(
        db_index=True,
        help_text="ID of the eScriptorium TaskReport associated with this job.",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="hpc_job_stats",
    )
    job_type = models.CharField(
        max_length=20,
        help_text="Type of training job: 'segtrain' or 'train'.",
    )
    slurm_job_id = models.CharField(max_length=50, blank=True)
    max_memory_mb = models.FloatField(
        null=True,
        blank=True,
        help_text="Peak memory usage in MB (MaxRSS from sacct).",
    )
    avg_memory_mb = models.FloatField(
        null=True,
        blank=True,
        help_text="Average memory usage in MB (AveRSS from sacct).",
    )
    elapsed_seconds = models.IntegerField(
        null=True,
        blank=True,
        help_text="Wall-clock runtime in seconds (ElapsedRaw from sacct).",
    )
    gpu_utilization = models.FloatField(
        null=True,
        blank=True,
        help_text="GPU utilization as a fraction 0-1 (from TRESUsageInTot gres/gpu).",
    )
    scratch_storage_mb = models.FloatField(
        null=True,
        blank=True,
        help_text="Total size of user's htr2hpc scratch directory in MB at job completion.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "HPC job stats"
        verbose_name_plural = "HPC job stats"
        ordering: ClassVar[list] = ["-created_at"]

    def __str__(self):
        return f"{self.user} {self.job_type} job {self.slurm_job_id or self.task_report_id}"
