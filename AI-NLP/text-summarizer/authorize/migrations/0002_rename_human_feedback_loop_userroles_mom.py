# Generated by Django 3.2.5 on 2022-03-09 05:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("authorize", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="userroles",
            old_name="human_feedback_loop",
            new_name="mom",
        ),
    ]
