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

from authorize.views import (AuthorizeNewUser,
                             GetAuthorizeuserSessionRrefreshToken,
                             GetAuthorizeuserSessionToken,
                             SetAuthorizeSessionProhibition)

urlpatterns = [
    path("newuser/", AuthorizeNewUser.as_view()),
    path(
        "login/",
        GetAuthorizeuserSessionToken.as_view(),
        name="Authorize session login",
    ),
    path(
        "refreshtoken/",
        GetAuthorizeuserSessionRrefreshToken.as_view(),
        name="Authorize refresh session toekn",
    ),
    path(
        "logout/",
        SetAuthorizeSessionProhibition.as_view(),
        name="Authorize session login",
    ),
]
