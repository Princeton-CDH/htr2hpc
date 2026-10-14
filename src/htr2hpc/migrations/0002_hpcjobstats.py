import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("htr2hpc", "0001_user_email_allow_null"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="HpcJobStats",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "task_report_id",
                    models.IntegerField(
                        db_index=True,
                        help_text="ID of the eScriptorium TaskReport associated with this job.",
                    ),
                ),
                (
                    "job_type",
                    models.CharField(
                        help_text="Type of training job: 'segtrain' or 'train'.",
                        max_length=20,
                    ),
                ),
                ("slurm_job_id", models.CharField(blank=True, max_length=50)),
                (
                    "max_memory_mb",
                    models.FloatField(
                        blank=True,
                        help_text="Peak memory usage in MB (MaxRSS from sacct).",
                        null=True,
                    ),
                ),
                (
                    "avg_memory_mb",
                    models.FloatField(
                        blank=True,
                        help_text="Average memory usage in MB (AveRSS from sacct).",
                        null=True,
                    ),
                ),
                (
                    "scratch_storage_mb",
                    models.FloatField(
                        blank=True,
                        help_text="Total size of user's htr2hpc scratch directory in MB at job completion.",
                        null=True,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="hpc_job_stats",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "HPC job stats",
                "verbose_name_plural": "HPC job stats",
                "ordering": ["-created_at"],
            },
        ),
    ]
