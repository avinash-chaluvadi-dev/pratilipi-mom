# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application urlpatterns***
    @Description    :
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""


from django.urls import path
from rest_framework import routers

from . import views

router = routers.DefaultRouter()
router.register("uploads", views.FileViewSet)
router.register("teams", views.TeamViewSet)


urlpatterns = [
    path("<model>/status/", views.status_check),
    path("file/status-update/", views.file_status_update),
    path("mom-data-filter/<request_id>/", views.data_filter),
    path("process-meeting/<request_id>/", views.process_meeting),
    path(
        "meeting-metadata/<request_id>/",
        views.MeetingMetadataRetrieveUpdateAPIView.as_view(),
    ),
    path(
        "mom/report/<request_id>/",
        views.GenerateMoMPDFAPIView.as_view(),
    ),
    path(
        "participants/",
        views.ConsolidateParticipantDashBoardBlenddatasets.as_view(),
        name="get",
    ),
    path(
        "create-issue/",
        views.ApiJiraTransactionView.as_view(),
        name="Create JIRA Issue",
    ),
    path(
        "participants/",
        views.ConsolidateParticipantDashBoardBlenddatasets.as_view(),
        name="get",
    ),
] + router.urls
