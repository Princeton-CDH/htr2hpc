from django.db import migrations

INITIAL_CONTENT = [
    {
        "key": "home_intro",
        "label": "Homepage intro text",
        "body": (
            "<p>A project providing digital recognition of handwritten documents"
            " using machine learning techniques.</p>\n"
            "<p>This is a limited-access eScriptorium instance managed by Princeton's"
            " Center for Digital Humanities to support affiliated research.</p>"
        ),
    },
    {
        "key": "contact_info",
        "label": "Contact page info",
        "body": "",
    },
    {
        "key": "models_list_intro",
        "label": "Models list description",
        "body": "",
    },
    {
        "key": "hpc_setup_intro",
        "label": "HPC setup instructions intro",
        "body": (
            "<p>To enable key-based ssh access to your account from the eScriptorium"
            " server, follow the below three steps. (This is a one-time process.)</p>"
        ),
    },
    {
        "key": "announcement",
        "label": "Site announcement",
        "body": "",
    },
]


def seed_initial_content(apps, schema_editor):
    SiteContent = apps.get_model("htr2hpc", "SiteContent")
    for item in INITIAL_CONTENT:
        SiteContent.objects.get_or_create(key=item["key"], defaults=item)


class Migration(migrations.Migration):
    dependencies = [
        ("htr2hpc", "0004_sitecontent"),
    ]

    operations = [
        migrations.RunPython(seed_initial_content, migrations.RunPython.noop),
    ]
