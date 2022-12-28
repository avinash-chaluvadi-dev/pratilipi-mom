# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Runtime IntakeEndpoint Initializer ***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

from . import yaml_model
from boiler_plate.utility.constants import APP_CONFIG_PATH

intake_endpoint = yaml_model.InputEndpoint(APP_CONFIG_PATH["input"])
