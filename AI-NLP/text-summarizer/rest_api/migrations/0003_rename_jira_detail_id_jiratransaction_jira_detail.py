# Generated by Django 3.2.5 on 2022-03-03 16:02

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("rest_api", "0002_auto_20220303_2131"),
    ]

    operations = [
        migrations.RenameField(
            model_name="jiratransaction",
            old_name="jira_detail_id",
            new_name="jira_detail",
        ),
    ]
