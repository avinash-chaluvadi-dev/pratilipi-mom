# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : OutputEndpoint Application Views ***
    @Description    : 
    @Date           : 25-08-2021
    @Last Modified  : 25-08-2021
"""

import logging as lg

from django.conf import settings
from rest_framework.response import Response
from rest_framework.views import APIView

from output_endpoint import output_endpoint

logger = lg.getLogger("file")


class UpdateOutputConfig(APIView):
    allowed_method = ("POST",)

    def post(self, request, *args, **kwargs):
        url = request.data["url"]
        # print("===============================>xxxxx::", url)
        output_endpoint.set_url(url)
        logger.info(f"Updated output endpoint with {url}")
        return Response({"message": "Ouput endpoint updated successfully"})
