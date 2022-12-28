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

from .models import ConsolidateModelsData, File, Team


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


# ConsolidateModelsData serializer
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

        def create(self, validated_data):
            ConsolidateModelsData = ConsolidateModelsData.objects.create(
                file_id=validated_data["file_id"],
                team_id=validated_data["team_id"],
                participant_id=validated_data["participant_id"],
                chunk_id=validated_data["chunk_id"],
                audio_path=validated_data["audio_path"],
                video_path=validated_data["video_path"],
                marker=validated_data["marker"],
                label=validated_data["label"],
                events=validated_data["events"],
            )
            return ConsolidateModelsData.objects.create(**validated_data)
