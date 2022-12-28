import asyncio
import logging as lg
from typing import Union

from django.conf import settings

from rest_api.models import File, MLModelStatus

from .http import async_get_requests, sync_get

logger = lg.getLogger("file")


def get_object_or_none(model: object, **kwargs: dict) -> Union[object, None]:
    try:
        return model.objects.get(**kwargs)
    except Exception:
        return None


def serve_ml_models(request_id: str):
    """This function calls all APIs for the available ML models

    Args:
        request_id: masked request id for uploaded meeting
    """

    file_obj = get_object_or_none(File, masked_request_id=request_id)

    logger.info(f"process started for meeting {request_id}")
    try:
        sd_response = sync_get(
            f"{settings.DIARIZATION_API_BASE_URL}module/speakerdiarization/{request_id}/"
        )
        sd_status = sd_response.status_code == 200 if sd_response is not None else False
        MLModelStatus.objects.update_or_create(
            file=file_obj,
            model_name="speaker_diarization",
            defaults={
                "status": sd_status,
            },
        )

        if sd_status:
            s2t_response = sync_get(
                f"{settings.S2T_API_BASE_URL}module/speechtotext/{request_id}/"
            )
            MLModelStatus.objects.update_or_create(
                file=file_obj,
                model_name="speechtotext",
                defaults={
                    "status": s2t_response.status_code == 200 if s2t_response else False
                },
            )

            framify_response = sync_get(
                f"{settings.KEYFRAME_API_BASE_URL}module/framify/{request_id}/",
            )
            MLModelStatus.objects.update_or_create(
                file=file_obj,
                model_name="framify",
                defaults={
                    "status": framify_response.status_code == 200
                    if framify_response
                    else False
                },
            )

        if get_object_or_none(
            MLModelStatus,
            model_name="speechtotext",
            status=True,
            file=file_obj,
        ):
            model_names = [
                "ner",
                "labelcls",
                "sentiment",
                "summarizer",
                "headliner",
                "esclation",
            ]
            urls = [
                f"{settings.CLASSIFIERS_API_BASE_URL}module/ner/{request_id}/",
                f"{settings.CLASSIFIERS_API_BASE_URL}module/labelcls/{request_id}/",
                f"{settings.CLASSIFIERS_API_BASE_URL}module/sentiment/{request_id}/",
                f"{settings.SUMMARIZER_API_BASE_URL}module/summarizer/{request_id}/",
                f"{settings.SUMMARIZER_API_BASE_URL}module/headliner/{request_id}/",
                f"{settings.CLASSIFIERS_API_BASE_URL}module/escalation/{request_id}/",
            ]

            if get_object_or_none(
                MLModelStatus, model_name="framify", status=True, file=file_obj
            ):
                model_names.append("keyframeext")
                urls.append(
                    f"{settings.KEYFRAME_API_BASE_URL}module/keyframeext/{request_id}/",
                )

            logger.info("starting async process")

            # Call the APIs asynchronously
            asyncio.run(async_get_requests(urls, model_names, file_obj))

        logger.info("writing in DB on success")
        file_obj.status = "Pending user review"
        file_obj.save()
        logger.info(f"process completed for meeting {request_id}")

    except Exception as e:
        logger.error(
            "An exception occured while processing request {}.".format(
                getattr(e, "message", repr(e))
            )
        )
        file_obj.status = "Pending user review"
        file_obj.save()
