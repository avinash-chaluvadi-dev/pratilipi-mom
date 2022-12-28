# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application Database Models***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""
import uuid

from django.db import models

from boiler_plate.utility import constants


def upload_to(instanc: object, file_name: str) -> str:
    """Returns the location where file will be uploaded"""
    location = f"{constants.MOM_UPLOAD_DIR}/{str(uuid.uuid4())}/{file_name}"
    return location


class File(models.Model):
    file = models.FileField(upload_to=upload_to, max_length=250)
    masked_request_id = models.CharField(max_length=250, blank=True, null=True)
    date = models.DateTimeField(auto_now=False, auto_now_add=True)
    backend_start_time = models.DateTimeField(
        auto_now=False, auto_now_add=False, blank=True, null=True
    )
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

    def get_file(self) -> object:
        return self.file

    def get_file_name(self) -> str:
        return self.file.name.split("/")[-1]

    def get_team_name(self) -> str:
        return self.team_name


class MeetingMetadata(models.Model):
    project_name = models.CharField(max_length=250, blank=True, null=True)
    mom_generation_date = models.DateField(
        auto_now=False, auto_now_add=False, blank=True, null=True
    )
    organiser = models.CharField(max_length=250, blank=True, null=True)
    location = models.CharField(max_length=250, blank=True, null=True)
    meeting_duration = models.CharField(max_length=50, blank=True, null=True)
    attendees = models.CharField(max_length=1000, blank=True, null=True)
    meeting = models.OneToOneField(
        "rest_api.File", on_delete=models.CASCADE, related_name="metadata"
    )

    class Meta:
        db_table = "api_meeting_metadata"
        ordering = ["-id"]


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


class TeamMember(models.Model):
    name = models.CharField(
        max_length=250,
    )
    email = models.CharField(max_length=250)
    team = models.ManyToManyField("rest_api.Team", related_name="team_members")

    class Meta:
        db_table = "api_team_member"
        ordering = ["-id"]


class JIRADetails(models.Model):
    ticket_no = models.CharField(max_length=250)
    team = models.OneToOneField(
        "rest_api.Team", on_delete=models.CASCADE, related_name="jira_details"
    )

    class Meta:
        db_table = "api_jira_details"
        ordering = ["-id"]


class FeedBackLoop(models.Model):
    request_id = models.CharField(max_length=250, blank=True)
    mom_output_v1 = models.FileField(upload_to=upload_to, max_length=250)
    mom_output_v2 = models.FileField(upload_to=upload_to, max_length=250)
    is_mom_exist = models.BooleanField(default=False)

    class Meta:
        db_table = "api_mom"
        ordering = ["-id"]

    def get_mom_output_v1(self) -> object:
        return self.mom_output_v1

    def get_mom_output_v2(self) -> object:
        return self.mom_output_v2


class ConsolidateModelsData(models.Model):
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


class UserModel(models.Model):
    user_id = models.IntegerField(primary_key=True)
    user_name = models.CharField(max_length=100)
    email = models.EmailField()
    team_id = models.IntegerField()
    role = models.CharField(max_length=50)

    class Meta:
        db_table = "api_user"


class JiraTransaction(models.Model):
    """Jira Configuration details
    {
        "id": "1280882",
        "key": "DEVVOTCJIR-9",
        "self": "https://jira-dev.anthem.com/rest/api/2/issue/1280882"
    }
    """

    jira_issue_id = models.CharField(max_length=50)
    jira_issue_key = models.CharField(max_length=250)
    jira_issue_url = models.CharField(max_length=250)
    jira_detail = models.ForeignKey(
        "rest_api.JIRADetails",
        on_delete=models.CASCADE,
    )
    timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)
    comments = models.CharField(max_length=250, blank=True, null=True)
    status = models.BooleanField(default=True)

    class Meta:
        managed = True
        db_table = "api_jira_transaction"
        ordering = ["-id"]


class MLModelStatus(models.Model):
    file = models.ForeignKey(File, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=200)
    status = models.BooleanField()

    class Meta:
        managed = True
        db_table = "api_ml_status"
        ordering = ["-id"]
