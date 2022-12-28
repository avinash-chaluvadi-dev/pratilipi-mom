# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : IntakeEndpoint Application ***
    @Description    : This utility class deals with all the YAML-related activity across the codebase. 
    @Date           : 25-08-2021
    @Last Modified  : 25-08-2021
"""

from django.apps import AppConfig


class IntakeEndpointConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "intake_endpoint"
