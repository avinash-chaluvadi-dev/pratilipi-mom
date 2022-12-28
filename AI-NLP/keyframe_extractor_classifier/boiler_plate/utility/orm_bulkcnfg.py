# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Global Constant file for Bulk ORM Datasets Insert in singe db transection.
                      This class keeps track of the ORM objects to be created for multiple requireds for the given
                      model classes and automatically creates action objects for bulk updates
                      when the number of objects accumulated for a given model class exceeds.
                      Finally we need to call the done() defincation to ensure to commit data into DB.***
    @Description    : 
    @Date           : 09-11-2021
    @Last Modified  : 09-11-2021
"""

from collections import defaultdict

from django.apps import apps


class SensemakerbulkDBMnger(object):
    def __init__(self, chunk_size=1000):
        self._create_queues = defaultdict(list)
        self.chunk_size = chunk_size

    def _commit(self, model_class):
        model_key = model_class._meta.label
        model_class.objects.bulk_create(self._create_queues[model_key])
        self._create_queues[model_key] = []

    def add(self, obj):
        model_class = type(obj)
        model_key = model_class._meta.label
        self._create_queues[model_key].append(obj)
        if len(self._create_queues[model_key]) >= self.chunk_size:
            self._commit(model_class)

    def done(self):
        for model_name, objs in self._create_queues.items():
            if len(objs) > 0:
                self._commit(apps.get_model(model_name))
