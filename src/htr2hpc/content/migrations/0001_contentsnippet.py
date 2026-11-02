from django.db import migrations, models
from wagtail.fields import RichTextField


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ContentSnippet",
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
                    "url_path",
                    models.CharField(
                        blank=True,
                        help_text=(
                            "URL path this snippet applies to (e.g. /contact/). "
                            "Leave blank for site-wide snippets (before_body, after_body)."
                        ),
                        max_length=200,
                    ),
                ),
                (
                    "slot",
                    models.CharField(
                        choices=[
                            ("intro", "Page intro"),
                            ("announcement", "Announcement banner"),
                            ("before_body", "Before body (site-wide)"),
                            ("after_body", "After body (site-wide)"),
                        ],
                        help_text="Where this snippet is injected on the page.",
                        max_length=100,
                    ),
                ),
                ("body", RichTextField()),
                (
                    "is_active",
                    models.BooleanField(
                        default=False,
                        help_text=(
                            "Only active snippets are displayed. "
                            "Toggle to publish or unpublish."
                        ),
                    ),
                ),
            ],
            options={
                "verbose_name": "content snippet",
                "verbose_name_plural": "content snippets",
                "ordering": ["url_path", "slot"],
            },
        ),
        migrations.AddConstraint(
            model_name="contentsnippet",
            constraint=models.UniqueConstraint(
                fields=["url_path", "slot"],
                name="unique_url_path_slot",
            ),
        ),
    ]
