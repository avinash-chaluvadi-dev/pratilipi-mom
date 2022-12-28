# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application Data Serializers***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""
import base64

from rest_framework import serializers

from .models import File, Team


class FileUploadSerializer(serializers.ModelSerializer):
    full_team_name = serializers.SerializerMethodField()

    def get_full_team_name(self, obj: object) -> str:
        return obj.team_name.name

    class Meta:
        model = File
        fields = (
            "id",
            "file",
            "masked_request_id",
            "date",
            "status",
            "team_name",
            "file_size",
            "full_team_name",
        )


class TeamSerializer(serializers.ModelSerializer):
    class Meta:
        model = Team
        fields = "__all__"
