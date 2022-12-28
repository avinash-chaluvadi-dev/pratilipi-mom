# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application views***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 22-01-2022
"""

import datetime
import json
import logging as lg
import os
import time
import uuid

from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.decorators import action, api_view
from rest_framework.generics import ListCreateAPIView
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from boiler_plate.utility.utils import handle_uploaded_files, model_status
from intake_endpoint import intake_endpoint
from output_endpoint import output_endpoint

from .models import (ConsolidateModelsData, File,
                     ParticipantMarkerCollaborationViewModel, Team)
from .serializers import (ConsolidateModelsDataSerializer,
                          FileUploadSerializer, TeamSerializer)

logger = lg.getLogger("file")


@api_view(["GET"])
def status_check(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper for checking the status of ML models
    """

    # Fetching model_name from kwargs
    model_name = kwargs.get("model")
    logger.info(f"Checking status of {model_name.capitalize()} job")

    response_dict = json.loads(request.body)
    response_path = response_dict[list(response_dict.keys())[0]]
    logger.info(f"Loading {model_name.capitalize()} output json from {response_path}")
    logger.info(f"{model_name.capitalize()} output json loaded successfully")

    status_response = model_status(file_path=response_path, model_name=model_name)
    logger.info(f"{model_name.capitalize()} status response --> {status_response}")

    return Response(status_response)


class FileViewSet(ModelViewSet):
    http_method_names = [
        "get",
        "post",
        "put",
        "patch",
        "head",
        "options",
        "trace",
    ]
    queryset = File.objects.exclude(status="cancelled")
    serializer_class = FileUploadSerializer

    def create(self, request: object, *args: list, **kwargs: dict) -> Response:
        """
        View used for uploading a new file
        """
        intake_url = request.data.get("intake_url")
        if intake_url is not None:
            intake_endpoint.set_url(intake_url)
        else:
            intake_endpoint.set_url(settings.MEDIA_ROOT)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data, status=status.HTTP_201_CREATED, headers=headers
        )

    @action(detail=True, methods=["PATCH"])
    def cancel_extraction(self, request: object, pk: int = None) -> Response:
        """
        This view will update the file status to cancelled
        """
        file = self.get_object()
        file.status = "cancelled"
        file.save()
        return Response(
            {"message": "File extraction cancelled"}, status=status.HTTP_200_OK
        )


class TeamListCreateAPIView(ListCreateAPIView):
    queryset = Team.objects.all()
    serializer_class = TeamSerializer


# consolidate-dataset-api:action-1
class ConsolidatedParticipantDashBoardDatasets(APIView):
    def get(self, request, *args, **kwargs) -> Response:
        try:
            from boiler_plate.utility.constants import \
                API_DATA_CONSOLIDATE_INDEXPOINTS
            from rest_api.models import File

            if File.objects.count() > 0:
                dictionary = {
                    "message": "The files uploaded data found !",
                    "status": True,
                    "tot_upc": File.objects.count(),
                    "tot_ror": File.objects.filter(
                        status=API_DATA_CONSOLIDATE_INDEXPOINTS["ror"]
                    ).count(),
                    "tot_mir": File.objects.filter(
                        status=API_DATA_CONSOLIDATE_INDEXPOINTS["mir"]
                    ).count(),
                    "tot_mgn": File.objects.filter(
                        status=API_DATA_CONSOLIDATE_INDEXPOINTS["mgn"]
                    ).count(),
                }
                # print(dictionary)
                response = {
                    "data": dictionary,
                    "success": True,
                    "status": status.HTTP_201_CREATED,
                    "message": "The files uploaded data found !",
                    "error": "",
                }
            else:
                response = {
                    "data": {},
                    "success": True,
                    "status": status.HTTP_201_CREATED,
                    "message": "No file uploads data found !",
                    "error": "",
                }
        except Exception as e:
            response = {
                "data": None,
                "success": False,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "The Exception participant window consolidate request...!",
                "error": "Something went wrong while processing participants window consolidate request...!",
            }
            logger.error(
                f"Something went wrong while processing participants window consolidate request...!\nstr{e}"
            )

        return Response(response)


# consolidate-dataset-api:action-2
class ConsolidateParticipantDashBoardBlenddatasets(generics.GenericAPIView):
    serializer_class = ConsolidateModelsDataSerializer

    def get(self, request, *args, **kwargs) -> Response:
        try:
            import re

            from django.db.models import Count, Sum

            from boiler_plate.utility import utils

            stime = time.strftime("%H:%M:%S:%MS", time.localtime())
            print("START CONSOLIDATING TIME ::", stime)

            # Get this user id from:auth+file+team
            obj_participants_consolidate_data = ConsolidateModelsData.objects.values(
                "participant_id"
            ).distinct()
            obj_consolidate_data = (
                ConsolidateModelsData.objects.values("file_id", "participant_id")
                .order_by("file_id")
                .annotate(Count("chunk_id"))
            )

            obj_mentoring_data = ConsolidateModelsData.objects.values(
                "file_id", "participant_id", "marker"
            ).filter(marker__contains="Mentoring And Engagement")
            obj_action_data = ConsolidateModelsData.objects.values(
                "file_id", "participant_id", "marker"
            ).filter(marker__contains="Action Plan Tracking")
            obj_collaboration_data = ConsolidateModelsData.objects.values(
                "file_id", "participant_id", "marker"
            ).filter(marker__contains="Collaboration")

            obj_proactiveness_data = ConsolidateModelsData.objects.values(
                "file_id", "participant_id", "marker"
            ).filter(marker__contains="Proactiveness")

            dtl_participant = {
                "tot_timelines": [],
                "tot_meetings": 0,
                "tot_collaboration": "",
                "tot_proactiveness": "",
                "tot_mentoring_engagement": "",
                "tot_action_plan_tracking": "",
            }

            tot_timelines = list()
            tot_meetings = 0
            tot_collaboration = 0
            tot_proactiveness = 0
            tot_action_plan_tracking = 0
            tot_mentoring_engagement = 0
            data = dict()
            collaboration_val = list()
            action_val = list()
            mentoring_val = list()
            proactiveness_val = list()

            for number_part in obj_participants_consolidate_data:

                # CONSOLIDATE:::CONSOLIDATE:::CONSOLIDATE
                queryset = obj_consolidate_data.filter(
                    participant_id=number_part.get("participant_id")
                )

                # MARKERS:::MARKERS:::MARKERS:::MARKERS
                query_mentoring_data = obj_mentoring_data.values("marker").filter(
                    participant_id=number_part.get("participant_id")
                )
                query_action_data = obj_action_data.values("marker").filter(
                    participant_id=number_part.get("participant_id")
                )
                query_collaboration_set = obj_collaboration_data.values(
                    "marker"
                ).filter(participant_id=number_part.get("participant_id"))
                query_proactiveness_data = obj_proactiveness_data.values(
                    "marker"
                ).filter(participant_id=number_part.get("participant_id"))

                for collaboration_marker in query_collaboration_set:
                    collaboration_val.append(
                        int(re.findall("[0-9]+", collaboration_marker.get("marker"))[0])
                    )
                # print("collaboration-sum::", sum(sorted(collaboration_val)))
                tot_collaboration = utils.get_marker_lable(
                    sum(sorted(collaboration_val))
                )

                for action_marker in query_action_data:
                    action_val.append(
                        int(re.findall("[0-9]+", action_marker.get("marker"))[0])
                    )
                # print("action-sum::", sum(sorted(action_val)))
                tot_action_plan_tracking = utils.get_marker_lable(
                    sum(sorted(action_val))
                )

                for mentoring_marker in query_mentoring_data:
                    mentoring_val.append(
                        int(re.findall("[0-9]+", mentoring_marker.get("marker"))[0])
                    )
                # print("mentoring-sum::", sum(sorted(mentoring_val)))
                tot_mentoring_engagement = utils.get_marker_lable(
                    sum(sorted(mentoring_val))
                )

                for proactiveness_marker in query_proactiveness_data:
                    proactiveness_val.append(
                        int(re.findall("[0-9]+", proactiveness_marker.get("marker"))[0])
                    )
                # print("proactiveness-sum::", sum(sorted(proactiveness_val)))
                tot_proactiveness = utils.get_marker_lable(
                    sum(sorted(proactiveness_val))
                )

                for item in queryset:
                    tot_timelines.append(item.get("chunk_id__count"))
                    tot_meetings += int(1)

                dtl_participant = {
                    "tot_timelines": tot_timelines,
                    "tot_meetings": tot_meetings,
                    "tot_collaboration": tot_collaboration,
                    "tot_proactiveness": tot_proactiveness,
                    "tot_mentoring_engagement": tot_mentoring_engagement,
                    "tot_action_plan_tracking": tot_action_plan_tracking,
                }

                data[number_part.get("participant_id")] = dtl_participant
                tot_timelines = list()
                tot_meetings = int(0)
                tot_collaboration = int(0)
                tot_proactiveness = int(0)
                tot_mentoring_engagement = int(0)
                tot_action_plan_tracking = int(0)
                dtl_participant = dict()

            # print("FINAL:", data)
            etime = time.strftime("%H:%M:%S:%MS", time.localtime())
            print("END CONSOLIDATING TIME ::", etime)

            response = {
                "data": data,
                "success": True,
                "status": status.HTTP_201_CREATED,
                "message": "The successful participant dataset ...!",
            }
            logger.info(f"The successful participant dataset ...!")

        except Exception as e:
            response = {
                "data": None,
                "success": False,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "The Exception while processing participants request ...!",
                "error": "Something went wrong while processing participants request...!",
            }
            logger.error(
                f"Something went wrong while processing participants request...!\nstr{e}"
            )

        return Response(response)
