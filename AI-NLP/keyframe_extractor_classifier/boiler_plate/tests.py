# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Model TestCases Shell Applications***
    @Description    : This is the generalized test case to validate other model related actions.
    @Date           : 25-08-2021
    @Last Modified  : 02-09-2021
"""
import logging as lg
import os
import re
from pathlib import Path

import yaml
from django.test import TestCase

logger = lg.getLogger("file")

"""
    1. Test case for shell consist proper number of applications or not
    2. Test case whether README.md exists or not
"""

cnfg_shell_apps = [
    "boiler_plate",
    "core_spec",
    "custom_rule_spec",
    "intake_endpoint",
    "logging_hook",
    "messaging",
    "module",
    "output_endpoint",
    "rest_api",
    "sense_maker",
    "storage_spec",
    "trigger",
]

cnfg_shell_listup = {
    "input": "cnfg_intake_endpoint.yml",
    "logging": "cnfg_logging_hook.yml",
    "messaging": "cnfg_messaging.yml",
    "output": "cnfg_output_endpoint.yml",
    "storage": "cnfg_storage_sepc.yml",
}
BASE_DIR = Path(__file__).resolve().parent.parent
appsList = [
    item
    for item in os.listdir(BASE_DIR)
    if (
        os.path.isdir(os.path.join(BASE_DIR, item))
        and os.path.exists(os.path.join(BASE_DIR, item, "__init__.py"))
    )
]

print("==========>>::", BASE_DIR)

app_folder = os.path.dirname(os.path.realpath(__file__))


def test_shellapps_exists():
    """test case to check the application exist or not"""
    error_list = []
    for index, appname in enumerate(appsList):
        if appname in cnfg_shell_apps[index]:
            pass
        else:
            error_list.append(appname)
    assert error_list == [], f"app is missing in the sense maker shell : {error_list}"


def test_readme_exists():
    """test case to check the application exist or not"""
    error_list = []
    for index, appname in enumerate(appsList):
        if appname in cnfg_shell_apps[index]:
            if os.path.exists("README.md"):
                pass
            else:
                error_list.append(appname)
    assert error_list == [], f"README.md file missing in the apps : {error_list}"
