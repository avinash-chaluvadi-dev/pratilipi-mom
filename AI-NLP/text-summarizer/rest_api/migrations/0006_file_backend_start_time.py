# Generated by Django 3.2.5 on 2022-11-16 01:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("rest_api", "0005_mlmodelstatus"),
    ]

    operations = [
        migrations.AddField(
            model_name="file",
            name="backend_start_time",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
