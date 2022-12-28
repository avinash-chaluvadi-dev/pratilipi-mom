# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : OutputEndpoint Application urlpatterns ***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 01-09-2021
"""

from django.urls import path

from . import views

urlpatterns = [
    path("mom/<request_id>/", views.MinutesOfMeetingAPIView.as_view()),
    path("summarizer/<request_id>/", views.text_summarizer_api_view),
    path("headliner/<request_id>/", views.headliner_generation),
]
