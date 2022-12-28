# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application views***
    @Description    : 
    @Date           : 07-10-2021
    @Last Modified  : 12-10-2021
"""

import logging as lg

from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView

from authorize.serializer import (AuthorizeNewUserRegisterSerializer,
                                  LoginSerializer)

logger = lg.getLogger("file")


# Authorize new user
class AuthorizeNewUser(generics.GenericAPIView):
    serializer_class = AuthorizeNewUserRegisterSerializer

    def post(self, request, *args, **kwargs):
        result = {"status": status.HTTP_201_CREATED}
        serializer = self.get_serializer(data=request.data)
        # print("data::serializer::",serializer)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        user.set_password(user.password)
        user.save()
        result["data"] = {
            "user": AuthorizeNewUserRegisterSerializer(
                user, context=self.get_serializer_context()
            ).data,
            "message": "New user has been created successfully !",
        }
        result["status"] = status.HTTP_201_CREATED

        return Response(status=result["status"], data=result["data"])


class LoginAPIView(APIView):
    serializer_class = LoginSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        return Response(serializer.data, status=status.HTTP_200_OK)
