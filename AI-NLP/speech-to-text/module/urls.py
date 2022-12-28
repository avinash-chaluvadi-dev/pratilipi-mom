# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : OutputEndpoint Application urlpatterns ***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

from django.urls import path

from . import views

urlpatterns = [
    path('speechtotext/<request_id>/', views.speech_to_text),

]
