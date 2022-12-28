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

from django.conf import settings
from django.db import models

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


class FeedBackLoop(models.Model):
    request_id = models.CharField(max_length=250, blank=True)
    mom_output_v1 = models.FileField(upload_to=upload_to, max_length=250)
    mom_output_v2 = models.FileField(upload_to=upload_to, max_length=250)
    # team_id = models.ForeignKey(
    #     "rest_api.Team", on_delete=models.DO_NOTHING, related_name="feedbackloopteam"
    # )
    # file_id = models.ForeignKey(
    #     "rest_api.File", on_delete=models.DO_NOTHING, related_name="feedbackloopfile"
    # )
    is_mom_exist = models.BooleanField(default=False)

    class Meta:
        db_table = "api_mom"
        ordering = ["-id"]

    def get_mom_output_v1(self) -> object:
        return self.mom_output_v1

    def get_mom_output_v2(self) -> object:
        return self.mom_output_v2


class ConsolidateModelsData(models.Model):
    """
    ------
    Info::
    ------
    This class model is pratilipi tool consolidate all the model data into single api.

    ----- ----- ------
    Table information
    ----- ----- ------
    :: cns_data_id file_id team_id participant_id chunk_id audio_path video_path marker label events cns_timestamp

    """

    cns_data_id = models.AutoField(primary_key=True, null=False, unique=True)
    file_id = models.CharField(max_length=10, blank=True, null=True)
    team_id = models.CharField(max_length=10, blank=True, null=True)
    participant_id = models.CharField(max_length=10, blank=True, null=True)
    chunk_id = models.CharField(max_length=10, blank=True, null=True)
    start_time = models.CharField(max_length=10, blank=True, null=True)
    end_time = models.CharField(max_length=10, blank=True, null=True)
    audio_path = models.CharField(max_length=1000, blank=True, null=True)
    video_path = models.CharField(max_length=1000, blank=True, null=True)
    marker = models.CharField(max_length=200, blank=True, null=True)
    label = models.CharField(max_length=200, blank=True, null=True)
    events = models.CharField(max_length=1000, blank=True, null=True)
    comments = models.CharField(max_length=250, blank=True, null=True)
    cns_timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)

    class Meta:
        db_table = "api_consolidate"


class ParticipantMarkerCollaborationViewModel(models.Model):
    cns_data_id = models.AutoField(primary_key=True, null=False, unique=True)
    file_id = models.CharField(max_length=10, blank=True, null=True)
    participant_id = models.CharField(max_length=10, blank=True, null=True)
    res_marker = models.CharField(max_length=200, blank=True, null=True)

    class Meta:
        managed = False
        db_table = "participant_marker_collaboration"
