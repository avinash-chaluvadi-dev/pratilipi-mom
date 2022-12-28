# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application signals***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""


from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import File, MeetingMetadata


@receiver(post_save, sender=File)
def insert_request_id(
    sender: object, instance: object, created: bool, **kwargs: dict
) -> None:
    """
    Create masked request ID
    """
    if created:
        instance.masked_request_id = "req_" + str(instance.pk).zfill(3)
        instance.save()
        MeetingMetadata.objects.create(meeting=instance)
