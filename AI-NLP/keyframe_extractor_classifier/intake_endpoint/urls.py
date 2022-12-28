# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : IntakeEndpoint Internal urlpatterns***
    @Description    : 
    @Date           : 25-08-2021
    @Last Modified  : 25-08-2021
"""

from django.urls import path

from . import views

urlpatterns = [
    path("update-input-endpoint/", views.UpdateInputConfig.as_view()),
]
