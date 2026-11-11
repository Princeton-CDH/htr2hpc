from django.db import migrations, models

# old slot -> new slot, for any snippets created with the first version of this app
SLOT_RENAMES = {
    "intro": "before_main",
    "announcement": "before_main",
    "before_body": "before_main",
    "after_body": "after_main",
}


def rename_slots(apps, schema_editor):
    ContentSnippet = apps.get_model("content", "ContentSnippet")
    # the homepage intro moves to its own slot
    ContentSnippet.objects.filter(url_path="/", slot="intro").update(
        slot="homepage_heading"
    )
    # the profile instructions are now shown after the main content
    ContentSnippet.objects.filter(url_path="/profile/", slot="intro").update(
        slot="after_main"
    )
    for old, new in SLOT_RENAMES.items():
        ContentSnippet.objects.filter(slot=old).update(slot=new)


class Migration(migrations.Migration):
    dependencies = [
        ("content", "0001_contentsnippet"),
    ]

    operations = [
        migrations.RenameField(
            model_name="contentsnippet",
            old_name="is_active",
            new_name="live",
        ),
        migrations.AlterField(
            model_name="contentsnippet",
            name="live",
            field=models.BooleanField(
                default=True,
                help_text="Only live snippets are displayed. Uncheck to hide without deleting.",
            ),
        ),
        migrations.AlterField(
            model_name="contentsnippet",
            name="url_path",
            field=models.CharField(
                blank=True,
                help_text=(
                    "URL for the page where this snippet should be displayed (e.g. /contact/). "
                    "Leave blank to display on all pages."
                ),
                max_length=200,
            ),
        ),
        migrations.AlterField(
            model_name="contentsnippet",
            name="slot",
            field=models.CharField(
                choices=[
                    ("before_main", "Before main content (any page)"),
                    ("after_main", "After main content (any page)"),
                    ("homepage_heading", "Homepage heading text"),
                ],
                help_text="Where on the page this snippet is displayed.",
                max_length=100,
            ),
        ),
        migrations.RunPython(rename_slots, migrations.RunPython.noop),
    ]
