# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Convert InputEndpoint YAML feed into django codebase object 
    @Description    : This utility class deals with all the YAML-related activity across the codebase. 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

import logging as lg

from boiler_plate.utility.read_yaml import ReadYaml

logger = lg.getLogger("file")


class InputEndpoint:
    def __init__(self, path):
        self.path = path
        yaml = ReadYaml(self.path)
        self.input_yaml = yaml.get_yaml()

    def get_applicable_config(self):
        return [
            s["parameters"]
            for s in self.input_yaml["intakeServiceDtls"]["execution"][
                "service_request"
            ]
            if s["applicable"]
        ][0]

    def get_input_yaml(self):
        return self.input_yaml

    def set_url(self, url):
        self.get_applicable_config()["url"] = url

    def get_url(self):
        return self.get_applicable_config()["url"]
