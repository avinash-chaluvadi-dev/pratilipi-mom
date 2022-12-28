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
router.register('uploads', views.FileViewSet)


urlpatterns = [
    path('teams/', views.TeamListCreateAPIView.as_view(), name='get-teams'),
] + router.urls
