# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application views***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

import os
import logging as lg
import uuid

from django.shortcuts import get_object_or_404
from django.conf import settings
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import ListCreateAPIView
from rest_framework.decorators import api_view
from rest_framework.viewsets import ModelViewSet
from rest_framework.decorators import action

from intake_endpoint import intake_endpoint
from boiler_plate.utility.utils import handle_uploaded_files
from .models import File, Team
from .serializers import FileUploadSerializer, TeamSerializer

logger = lg.getLogger("file")


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
