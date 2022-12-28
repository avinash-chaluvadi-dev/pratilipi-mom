# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application Database Models***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""
import uuid
from pathlib import Path

from django.db import models
from django.conf import settings

from intake_endpoint import intake_endpoint


def upload_to(instanc: object, file_name: str) -> str:
    """Returns the location where file will be uploaded"""
    location = f"{str(uuid.uuid4())}/{file_name}"
    if not settings.MEDIA_ROOT == intake_endpoint.get_url():
        location = f"{intake_endpoint.get_url()}/{location}"

    return location


class File(models.Model):
    file = models.FileField(upload_to=upload_to, max_length=250)
    masked_request_id = models.CharField(max_length=250, blank=True, null=True)
    date = models.DateField(auto_now=False, auto_now_add=True)
    status = models.CharField(max_length=50, default="Uploaded")
    team_name = models.ForeignKey(
        "rest_api.Team", on_delete=models.DO_NOTHING, related_name="files"
    )
    file_size = models.CharField(max_length=250, blank=True, null=True)

    class Meta:
        db_table = "api_file"
        ordering = ["-id"]

    def get_file(self) -> object:
        return self.file

    def get_file_name(self) -> str:
        return self.file.name.split("/")[-1]


class Team(models.Model):
    name = models.CharField(max_length=250)

    class Meta:
        db_table = "api_team"
        ordering = ["-id"]

    def __str__(self):
        return self.name
