from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("htr2hpc", "0003_hpcjobstats_elapsed_gpu"),
    ]

    operations = [
        migrations.CreateModel(
            name="SiteContent",
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
                ("key", models.SlugField(max_length=100, unique=True)),
                ("label", models.CharField(max_length=200)),
                (
                    "body",
                    models.TextField(
                        blank=True,
                        help_text="HTML content to inject into the page. Leave blank to hide the slot.",
                    ),
                ),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "site content",
                "verbose_name_plural": "site content",
                "ordering": ["key"],
            },
        ),
    ]
