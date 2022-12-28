import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.response import Response

from boiler_plate.utility import constants
from boiler_plate.utility.utils import (S3Utils, create_error_response,
                                        get_file_base_path,
                                        get_input_file_as_dict,
                                        write_to_output_location)
from module import specs
from module.Framify import model_serve as framify_serve
from module.image_classifier import model_serve as keyframe_classifier_serve
from rest_api.models import File

from .utils import handle_failure, scheduler_config

logger = logging.getLogger("file")


def background_run(**kwargs: dict):
    model = kwargs.get("model")
    scheduler = kwargs.get("scheduler")
    input_file = kwargs.get("input_file")
    response_path = kwargs.get("response_path")
    serve_function = kwargs.get("serve_function")

    if model == "keyframe_classifier":
        # Calling the ML serve function
        output = serve_function(input_file)
        write_to_output_location(output_path=response_path, output=output)
    else:
        output_path = str(Path(response_path).parent) + "/"
        # Calling the ML serve function
        output = serve_function(input_file, output_path)
        write_to_output_location(output_path=response_path, output=output)


def background_listener(event, **kwargs: dict):
    # Get keyword arguments which further passed to scheduler object
    model = kwargs.get("kwargs").get("model")
    scheduler = kwargs.get("kwargs").get("scheduler")
    request_id = kwargs.get("kwargs").get("request_id")
    response_path = kwargs.get("kwargs").get("response_path")

    # Creates model object to interact with database table
    file_obj = get_object_or_404(File, masked_request_id=request_id)

    if event.code == constants.EVENT_SUCCESS:
        logger.info(f"{scheduler} job executed successfully")
        logger.info("Started killing the running thread/process")
        globals()[scheduler].shutdown(wait=False)
        logger.info("Running thread/process killed successfully")
        del globals()[scheduler]

    if event.code == constants.EVENT_FAIL:
        logger.info(
            f"{scheduler} job has been suddenly crashed due to runtime/memory error"
        )
        handle_failure(
            model_name=model,
            file_obj=file_obj,
            request_id=request_id,
            response_path=response_path,
        )
        logger.info("Started killing the running thread/process")
        globals()[scheduler].shutdown(wait=False)
        logger.info("Running thread/process killed successfully")
        del globals()[scheduler]


@api_view(["GET"])
def keyframe_extraction(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper over Keyframe Extractor (Framify) Model
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"keyframe_ext_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating keyframe specs from specs.yaml file
    framify_specs = specs.get_framify_specs()
    framify_input = framify_specs.get("input")
    input_file_name = framify_specs.get("input_file_name")
    framify_output = framify_specs.get("output")
    output_file_name = framify_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        framify_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass to serve function
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    framify_response_path = f"{output_base_path}{framify_output}/{output_file_name}"
    response_path = f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{framify_response_path}"

    # Loading input file as dict to check status of S2T/Diarization
    output_exist = s3_utils.prefix_exist(file_prefix=input_file_path)

    input_file = get_input_file_as_dict(
        file_path=input_file_path, output_exist=output_exist
    )

    # Get's status from api_file table
    db_status = file_obj.status
    if (
        db_status == constants.ERROR_DB
        or not output_exist
        or input_file.get(constants.STATUS_KEY) == constants.ERROR_KEY
    ):
        # Calls handle_failure function to update headliner status to error
        handle_failure(
            file_obj=file_obj,
            request_id=request_id,
            model_name=constants.FRAMIFY,
            response_path=framify_response_path,
        )

        # Returns error response
        response = {constants.RESPONSE[constants.FRAMIFY]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.FRAMIFY],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(
        output_path=framify_response_path, output=progress_response
    )

    # Initializing kwargs for background process and listener
    listener_dict = {
        "request_id": request_id,
        "model": constants.FRAMIFY,
        "scheduler": scheduler_obj_name,
        "response_path": framify_response_path,
    }
    param_dict = {
        "input_file": input_file,
        "scheduler": scheduler_obj_name,
        "serve_function": framify_serve,
        "response_path": framify_response_path,
        "model": constants.MODEL_NAMES[constants.FRAMIFY],
    }
    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.FRAMIFY]: response_path}
    return Response(response)


@api_view(["GET"])
def keyframe_classifier(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper over Keyframe Extraction Classifierr Model
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"keyframe_cls_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating keyframe specs from specs.yaml file
    keyframe_cls_specs = specs.get_keyframe_cls_specs()
    keyframe_cls_input = keyframe_cls_specs.get("input")
    input_file_name = keyframe_cls_specs.get("input_file_name")
    keyframe_cls_output = keyframe_cls_specs.get("output")
    output_file_name = keyframe_cls_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        keyframe_cls_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass extractor model
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    keyframe_cls_response_path = (
        f"{output_base_path}{keyframe_cls_output}/{output_file_name}"
    )
    response_path = (
        f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{keyframe_cls_response_path}"
    )

    # Loading input file as dict to check status of S2T/Diarization
    output_exist = s3_utils.prefix_exist(file_prefix=input_file_path)
    input_file = get_input_file_as_dict(
        file_path=input_file_path, output_exist=output_exist
    )

    # Get's status from api_file table
    db_status = file_obj.status
    if (
        db_status == constants.ERROR_DB
        or not output_exist
        or input_file.get(constants.STATUS_KEY) == constants.ERROR_KEY
    ):
        # Calls handle_failure function to update headliner status to error
        handle_failure(
            file_obj=file_obj,
            request_id=request_id,
            model_name=constants.KEYFRAME_CLS,
            response_path=keyframe_cls_response_path,
        )

        # Returns error response
        response = {constants.RESPONSE[constants.KEYFRAME_CLS]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.KEYFRAME_CLS],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(
        output_path=keyframe_cls_response_path, output=progress_response
    )

    # Initializing kwargs for background process and listener
    listener_dict = {
        "request_id": request_id,
        "model": constants.KEYFRAME_CLS,
        "scheduler": scheduler_obj_name,
        "response_path": keyframe_cls_response_path,
    }
    param_dict = {
        "input_file": input_file,
        "scheduler": scheduler_obj_name,
        "response_path": keyframe_cls_response_path,
        "serve_function": keyframe_classifier_serve,
        "model": constants.MODEL_NAMES[constants.KEYFRAME_CLS],
    }

    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.KEYFRAME_CLS]: response_path}
    return Response(response)
