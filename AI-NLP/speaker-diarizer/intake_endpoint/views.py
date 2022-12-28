# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : IntakeEndpoint Standard Views***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
import logging as lg

from intake_endpoint import intake_endpoint

logger = lg.getLogger("file")


class UpdateIntakeConfig(APIView):
    allowed_method = ("POST",)

    def post(self, request, *args, **kwargs):
        url = request.data["url"]
        intake_endpoint.set_url(url)
        logger.info(f"Updated Intake endpoint with {url}")
        return Response({"message": "Intake location updated successfully"})
