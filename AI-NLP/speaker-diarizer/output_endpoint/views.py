# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : OutputEndpoint Application Views ***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
import logging as lg

from output_endpoint import output_endpoint

logger = lg.getLogger("file")


class UpdateOutputConfig(APIView):
    allowed_method = ("POST",)

    def post(self, request, *args, **kwargs):
        url = request.data["url"]
        output_endpoint.set_url(url)
        logger.info(f"Updated output endpoint with {url}")
        return Response({"message": "Ouput endpoint updated successfully"})
