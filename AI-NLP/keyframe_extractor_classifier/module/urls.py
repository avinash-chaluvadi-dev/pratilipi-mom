# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Model Application urlpatterns ***
    @Description    : 
    @Date           : 25-08-2021
    @Last Modified  : 25-08-2021
"""

from django.urls import path

from module.views import keyframe_classifier, keyframe_extraction

urlpatterns = [
    path("framify/<request_id>/", keyframe_extraction),
    path("keyframecls/<request_id>/", keyframe_classifier),
]
