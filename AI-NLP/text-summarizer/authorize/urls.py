# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application views***
    @Description    : 
    @Date           : 07-10-2021
    @Last Modified  : 12-10-2021
"""

from django.conf.urls import url
from django.urls import include, path

from authorize.views import AuthorizeNewUser, LoginAPIView

urlpatterns = [
    path("newuser/", AuthorizeNewUser.as_view()),
    path(
        "login/",
        LoginAPIView.as_view(),
        name="Authorize session login",
    ),
]
