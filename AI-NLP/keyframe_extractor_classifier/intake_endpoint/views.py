# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : IntakeEndpoint Standard Views***
    @Description    : 
    @Date           : 25-08-2021
    @Last Modified  : 25-08-2021
"""

import logging as lg

from django.conf import settings
from rest_framework.response import Response
from rest_framework.views import APIView

from intake_endpoint import intake_endpoint

logger = lg.getLogger("file")


class UpdateInputConfig(APIView):
    allowed_method = ("POST",)

    def post(self, request, *args, **kwargs):
        url = request.data["url"]
        # print("===============================>xxxxx::", url)
        intake_endpoint.set_url(url)
        logger.info(f"Updated Input endpoint with {url}")
        return Response({"message": "Input location updated successfully"})
