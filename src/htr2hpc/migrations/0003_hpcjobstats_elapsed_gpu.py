from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("htr2hpc", "0002_hpcjobstats"),
    ]

    operations = [
        migrations.AddField(
            model_name="hpcjobstats",
            name="elapsed_seconds",
            field=models.IntegerField(
                blank=True,
                help_text="Wall-clock runtime in seconds (ElapsedRaw from sacct).",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="hpcjobstats",
            name="gpu_utilization",
            field=models.FloatField(
                blank=True,
                help_text="GPU utilization as a fraction 0-1 (from TRESUsageInTot gres/gpu).",
                null=True,
            ),
        ),
    ]
