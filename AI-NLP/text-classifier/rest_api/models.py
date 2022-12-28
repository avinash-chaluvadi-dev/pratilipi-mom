# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application Database Models***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""
import uuid

from django.conf import settings
from django.db import models


def upload_to(instanc: object, file_name: str) -> str:
    """Returns the location where file will be uploaded"""
    location = f"{str(uuid.uuid4())}/{file_name}"

    return location


class File(models.Model):
    file = models.FileField(upload_to=upload_to, max_length=250)
    masked_request_id = models.CharField(max_length=250, blank=True, null=True)
    date = models.DateField(auto_now=False, auto_now_add=True)
    status = models.CharField(max_length=50, default="Uploaded")
    team_name = models.ForeignKey(
        "rest_api.Team", on_delete=models.DO_NOTHING, related_name="files"
    )
    user = models.ForeignKey(
        "authorize.CustomUser",
        on_delete=models.DO_NOTHING,
        related_name="files",
    )
    file_size = models.CharField(max_length=250, blank=True, null=True)

    class Meta:
        db_table = "api_file"
        ordering = ["-id"]
        managed = False

    def get_file(self) -> object:
        return self.file

    def get_file_name(self) -> str:
        return self.file.name.split("/")[-1]

    def get_team_name(self) -> str:
        return self.team_name


class Team(models.Model):
    name = models.CharField(max_length=250, unique=True)
    dl_email = models.CharField(max_length=250)
    sme_name = models.CharField(max_length=250)
    sme_email = models.CharField(max_length=250)
    sme_email_notification = models.BooleanField(default=True)
    po_name = models.CharField(max_length=250)
    po_email = models.CharField(max_length=250)
    po_email_notification = models.BooleanField(default=False)
    manager_name = models.CharField(max_length=250)
    manager_email = models.CharField(max_length=250)
    manager_email_notification = models.BooleanField(default=False)
    created_date = models.DateField(auto_now=False, auto_now_add=True)

    class Meta:
        db_table = "api_team"
        ordering = ["-id"]

    def __str__(self):
        return self.name
