# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application Data Serializers***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""
from datetime import date

from django.db import transaction
from rest_framework import serializers

from .models import (ConsolidateModelsData, File, JIRADetails, JiraTransaction,
                     MeetingMetadata, Team, TeamMember)


class FileUploadSerializer(serializers.ModelSerializer):
    full_team_name = serializers.SerializerMethodField()

    def get_full_team_name(self, obj: object) -> str:
        return obj.team_name.name

    def create(self, validated_data):
        self.context["request"].user
        return File.objects.create(user=self.context["request"].user, **validated_data)

    def update(self, instance: object, validated_data: dict) -> object:
        """Overriding update method for serialization"""
        if (
            validated_data.get("status") == "User_review_inprogress"
            and not instance.metadata.mom_generation_date
        ):
            instance.metadata.mom_generation_date = date.today()
            instance.metadata.save()
        return super(FileUploadSerializer, self).update(instance, validated_data)

    class Meta:
        model = File
        fields = (
            "id",
            "date",
            "file",
            "status",
            "file_size",
            "team_name",
            "full_team_name",
            "masked_request_id",
            "backend_start_time",
        )


class FileTeamNameSerializer(serializers.ModelSerializer):
    full_team_name = serializers.SerializerMethodField()

    def get_full_team_name(self, obj: object) -> str:
        return obj.team_name.name

    class Meta:
        model = File
        fields = (
            "team_name",
            "full_team_name",
        )


class MeetingMetadataSerializer(serializers.ModelSerializer):
    meeting = FileTeamNameSerializer()
    uploaded_date = serializers.SerializerMethodField()
    meeting_status = serializers.SerializerMethodField()

    def get_uploaded_date(self, obj):
        return obj.meeting.date

    def get_meeting_status(self, obj):
        return obj.meeting.status

    class Meta:
        model = MeetingMetadata
        fields = (
            "project_name",
            "mom_generation_date",
            "organiser",
            "location",
            "meeting_duration",
            "attendees",
            "meeting",
            "uploaded_date",
            "meeting_status",
        )

    def update(self, instance: object, validated_data: dict) -> object:
        """Overriding update method for serialization"""
        if "meeting" in validated_data:
            meeting = validated_data.pop("meeting")
            File.objects.filter(pk=instance.meeting.pk).update(
                team_name=meeting["team_name"]
            )
        return super(MeetingMetadataSerializer, self).update(instance, validated_data)


class TeamMemberSerializer(serializers.ModelSerializer):
    class Meta:
        model = TeamMember
        exclude = ("team",)


class JIRADetailsSerializer(serializers.ModelSerializer):
    class Meta:
        model = JIRADetails
        fields = ("ticket_no",)


class TeamSerializer(serializers.ModelSerializer):
    team_members = TeamMemberSerializer(many=True, required=False)
    jira_details = JIRADetailsSerializer()

    class Meta:
        model = Team
        fields = (
            "id",
            "name",
            "dl_email",
            "sme_name",
            "sme_email",
            "sme_email_notification",
            "po_name",
            "po_email",
            "po_email_notification",
            "manager_name",
            "manager_email",
            "manager_email_notification",
            "created_date",
            "team_members",
            "jira_details",
        )
        extra_kwrags = {"created_date": {"read_only": True}}

    def create(self, validated_data):
        with transaction.atomic():
            members_data = []
            if "team_members" in validated_data:
                members_data = validated_data.pop("team_members")
            jira_data = validated_data.pop("jira_details")
            team = Team.objects.create(**validated_data)
            JIRADetails.objects.create(team=team, **jira_data)
            for member_data in members_data:
                member, _ = TeamMember.objects.get_or_create(**member_data)
                member.team.add(team)
        return team

    def update(self, instance, validated_data):
        with transaction.atomic():
            if "team_members" in validated_data:
                details_in_db = TeamMember.objects.filter(team=instance)
                members_data = validated_data.pop("team_members")

                # Remove  team members
                ids_to_keep = [x["id"] for x in members_data if x.get("id") is not None]
                for d in details_in_db:
                    if d.id not in ids_to_keep:
                        d.team.remove(instance)

                # Add Team members
                for member_data in members_data:
                    member, _ = TeamMember.objects.get_or_create(**member_data)
                    member.team.add(instance)

            if "jira_details" in validated_data:
                jira_data = validated_data.pop("jira_details")
                JIRADetailsSerializer().update(instance.jira_details, jira_data)

            super().update(instance, validated_data)
        return instance


class ConsolidateModelsDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConsolidateModelsData
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
        fields = (
            "file_id",
            "team_id",
            "participant_id",
            "chunk_id",
            "audio_path",
            "video_path",
            "marker",
            "label",
            "events",
        )


class JiraTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = JiraTransaction
        fields = "__all__"
