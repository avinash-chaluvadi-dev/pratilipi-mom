# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Global Application Utility YAML parser***
    @Description    : This is the generalized codebase to convert YAML to django object.
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""
import yaml
import logging as lg

logger = lg.getLogger('file')


class ReadYaml():
    """
    A Generic class which takes in yaml and converts it into a python object
    """

    def __init__(self, path):
        self.path = path

    def get_yaml(self):
        try:
            with open(self.path) as stream:
                data = yaml.safe_load(stream)
                logger.info('testing logging')
                return data
        except Exception as e:
            logger.info('testing logging')
            print(str(e))

    def get_applicable_log_config(self):
        return [s['parameters'] for s in self.get_yaml()['logServiceDtls']['execution']['service_request'] if s['applicable']][0]
