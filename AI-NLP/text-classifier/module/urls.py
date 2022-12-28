# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : OutputEndpoint Application urlpatterns ***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 14-10-2021
"""

from django.urls import path

from . import views

urlpatterns = [
    path("ner/<request_id>/", views.ner_api_view),
    path("allocator/<request_id>/", views.allocator_api_view),
    path("markercls/<request_id>/", views.marker_api_view),
    path("labelcls/<request_id>/", views.label_api_view),
    path("markercls/<request_id>/", views.marker_api_view),
    path("sentiment/<request_id>/", views.sentiment_api_view),
    path("escalation/<request_id>/", views.escalation_api_view),
]
