# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Global Application SETUP config***
    @Description    : This is the generalized codebase to convert YAML to django object.
    @Date           : 25-08-2021
    @Last Modified  : 25-08-2021
"""
from boiler_plate.utility.read_yaml import ReadYaml


class Specs:
    def __init__(self, path):
        self.path = path
        yaml = ReadYaml(self.path)
        self.specs = yaml.get_yaml()

    def get_framify_specs(self):
        return self.specs["framify"]

    def get_keyframe_cls_specs(self):
        return self.specs["keyframe_classifier"]
