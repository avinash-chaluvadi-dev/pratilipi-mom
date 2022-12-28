# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : IntakeEndpoint Internal urlpatterns***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

from django.urls import path

from . import views

urlpatterns = [
    path("update-intake-endpoint/", views.UpdateIntakeConfig.as_view()),
]
