# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Logging Application ***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

from django.apps import AppConfig


class LoggingHookConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "logging_hook"
