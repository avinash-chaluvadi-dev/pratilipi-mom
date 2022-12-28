# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Convert OutputEndpoint YAML feed into django codebase object 
    @Description    : This utility class deals with all the YAML-related activity across the codebase. 
    @Date           : 25-08-2021
    @Last Modified  : 25-08-2021
"""

import logging as lg

import yaml

from boiler_plate.utility.read_yaml import ReadYaml

logger = lg.getLogger("file")


class OutputEndpoint:
    def __init__(self, path):
        self.path = path
        yaml = ReadYaml(self.path)
        self.output_yaml = yaml.get_yaml()

    def get_applicable_config(self):
        return [
            s["parameters"]
            for s in self.output_yaml["outputServiceDtls"]["execution"][
                "service_request"
            ]
            if s["applicable"]
        ][0]

    def get_output_yaml(self):
        return self.output_yaml

    def set_url(self, url):
        self.get_applicable_config()["url"] = url

    def get_url(self):
        return self.get_applicable_config()["url"]
