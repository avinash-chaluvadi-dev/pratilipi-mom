# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Runtime IntakeEndpoint Initializer ***
    @Description    : 
    @Date           : 25-08-2021
    @Last Modified  : 25-08-2021
"""

from boiler_plate.utility.constants import APP_CONFIG_PATH

from . import yaml_model

intake_endpoint = yaml_model.InputEndpoint(APP_CONFIG_PATH["input"])
