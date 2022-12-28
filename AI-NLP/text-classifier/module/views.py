import logging as lg
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
from boiler_plate.utility.utils import (S3Utils, get_file_base_path,
                                        get_input_file_as_dict,
                                        write_to_output_location)
from module import specs
from rest_api.models import File

from .Allocator import model_serve as allocator_model
from .escalation_classifier import main as escalation_model
from .label_classifier import model_serve as label_model
from .marker_classifier import model_serve as marker_model
from .SentimentClassifier import model_serve as sentiment_model
from .spacy_ner import model_serve as ner_model
from .utils import handle_failure, scheduler_config

logger = lg.getLogger("file")
# Creates s3_utils object to interact with S3 using boto3
S3_UTILS = S3Utils()


def background_run(**kwargs: dict):
    model = kwargs.get("model")
    scheduler = kwargs.get("scheduler")
    input_file = kwargs.get("input_file")
    response_path = kwargs.get("response_path")
    serve_function = kwargs.get("serve_function")

    # Calling the ML serve function
    output = serve_function(input_file)
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
def ner_api_view(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper over NER Model
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"ner_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating ner specs from specs.yaml file
    ner_specs = specs.get_ner_specs()
    ner_input = ner_specs.get("input")
    input_file_name = ner_specs.get("input_file_name")
    ner_output = ner_specs.get("output")
    output_file_name = ner_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        ner_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass to serve function
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    ner_response_path = f"{output_base_path}{ner_output}/{output_file_name}"
    response_path = f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{ner_response_path}"

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
        # Calls handle_failure function to update NER status to error
        handle_failure(
            file_obj=file_obj,
            request_id=request_id,
            model_name=constants.NER,
            response_path=ner_response_path,
        )

        # Returns error response
        response = {constants.RESPONSE[constants.NER]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.NER],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(output_path=ner_response_path, output=progress_response)

    # Initializing kwargs for background process and listener
    listener_dict = {
        "model": constants.NER,
        "request_id": request_id,
        "scheduler": scheduler_obj_name,
        "response_path": ner_response_path,
    }
    param_dict = {
        "input_file": input_file,
        "serve_function": ner_model,
        "scheduler": scheduler_obj_name,
        "response_path": ner_response_path,
        "model": constants.MODEL_NAMES[constants.NER],
    }
    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.NER]: response_path}
    return Response(response)


@api_view(["GET"])
def sentiment_api_view(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper over Sentiment Classifier Model
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"sentiment_cls_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating sentiment specs from specs.yaml file
    sentiment_specs = specs.get_sentiment_specs()
    sentiment_input = sentiment_specs.get("input")
    input_file_name = sentiment_specs.get("input_file_name")
    sentiment_output = sentiment_specs.get("output")
    output_file_name = sentiment_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        sentiment_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass to serve function
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    sentiment_response_path = f"{output_base_path}{sentiment_output}/{output_file_name}"
    response_path = f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{sentiment_response_path}"

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
        # Calls handle_failure function to update NER status to error
        handle_failure(
            file_obj=file_obj,
            request_id=request_id,
            model_name=constants.SENTIMENT_CLS,
            response_path=sentiment_response_path,
        )

        # Returns error response
        response = {constants.RESPONSE[constants.SENTIMENT_CLS]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.SENTIMENT_CLS],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(
        output_path=sentiment_response_path, output=progress_response
    )

    # Initializing kwargs for background process and listener
    listener_dict = {
        "request_id": request_id,
        "scheduler": scheduler_obj_name,
        "model": constants.SENTIMENT_CLS,
        "response_path": sentiment_response_path,
    }
    param_dict = {
        "input_file": input_file,
        "scheduler": scheduler_obj_name,
        "serve_function": sentiment_model,
        "response_path": sentiment_response_path,
        "model": constants.MODEL_NAMES[constants.SENTIMENT_CLS],
    }
    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.SENTIMENT_CLS]: response_path}
    return Response(response)


@api_view(["GET"])
def label_api_view(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper over Label Model
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"label_cls_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating label specs from specs.yaml file
    label_specs = specs.get_label_specs()
    label_input = label_specs.get("input")
    input_file_name = label_specs.get("input_file_name")
    label_output = label_specs.get("output")
    output_file_name = label_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        label_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass to serve function
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    label_response_path = f"{output_base_path}{label_output}/{output_file_name}"
    response_path = f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{label_response_path}"

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
        # Calls handle_failure function to update NER status to error
        handle_failure(
            file_obj=file_obj,
            request_id=request_id,
            model_name=constants.LABEL_CLS,
            response_path=label_response_path,
        )

        # Returns error response
        response = {constants.RESPONSE[constants.LABEL_CLS]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.LABEL_CLS],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(output_path=label_response_path, output=progress_response)

    # Initializing kwargs for background process and listener
    listener_dict = {
        "request_id": request_id,
        "model": constants.LABEL_CLS,
        "scheduler": scheduler_obj_name,
        "response_path": label_response_path,
    }
    param_dict = {
        "input_file": input_file,
        "serve_function": label_model,
        "scheduler": scheduler_obj_name,
        "response_path": label_response_path,
        "model": constants.MODEL_NAMES[constants.LABEL_CLS],
    }
    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.LABEL_CLS]: response_path}
    return Response(response)


@api_view(["GET"])
def marker_api_view(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper over Marker Model
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"marker_cls_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating marker specs from specs.yaml file
    marker_specs = specs.get_marker_specs()
    marker_input = marker_specs.get("input")
    input_file_name = marker_specs.get("input_file_name")
    marker_output = marker_specs.get("output")
    output_file_name = marker_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        marker_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass to serve function
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    marker_response_path = f"{output_base_path}{marker_output}/{output_file_name}"
    response_path = f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{marker_response_path}"

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
        # Calls handle_failure function to update NER status to error
        handle_failure(
            file_obj=file_obj,
            request_id=request_id,
            model_name=constants.MARKER_CLS,
            response_path=marker_response_path,
        )

        # Returns error response
        response = {constants.RESPONSE[constants.MARKER_CLS]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.MARKER_CLS],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(output_path=marker_response_path, output=progress_response)

    # Initializing kwargs for background process and listener
    listener_dict = {
        "request_id": request_id,
        "model": constants.MARKER_CLS,
        "scheduler": scheduler_obj_name,
        "response_path": marker_response_path,
    }
    param_dict = {
        "input_file": input_file,
        "serve_function": marker_model,
        "scheduler": scheduler_obj_name,
        "response_path": marker_response_path,
        "model": constants.MODEL_NAMES[constants.MARKER_CLS],
    }
    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.MARKER_CLS]: response_path}
    return Response(response)


@api_view(["GET"])
def allocator_api_view(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper over Deadline Escalation Model
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"deadline_cls_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating allocator specs from specs.yaml file
    allocator_specs = specs.get_allocator_specs()
    allocator_input = allocator_specs.get("input")
    input_file_name = allocator_specs.get("input_file_name")
    allocator_output = allocator_specs.get("output")
    output_file_name = allocator_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        allocator_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass to serve function
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    allocator_response_path = f"{output_base_path}{allocator_output}/{output_file_name}"
    response_path = f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{allocator_response_path}"

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
        # Calls handle_failure function to update summarizer status to error
        handle_failure(
            file_obj=file_obj,
            request_id=request_id,
            model_name=constants.ALLOCATOR_CLS,
            response_path=allocator_response_path,
        )

        # Returns error response
        response = {constants.RESPONSE[constants.ALLOCATOR_CLS]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.ALLOCATOR_CLS],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(
        output_path=allocator_response_path, output=progress_response
    )

    # Initializing kwargs for background process and listener
    listener_dict = {
        "request_id": request_id,
        "scheduler": scheduler_obj_name,
        "model": constants.ALLOCATOR_CLS,
        "response_path": allocator_response_path,
    }
    param_dict = {
        "input_file": input_file,
        "scheduler": scheduler_obj_name,
        "serve_function": allocator_model,
        "response_path": allocator_response_path,
        "model": constants.MODEL_NAMES[constants.ALLOCATOR_CLS],
    }
    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.ALLOCATOR_CLS]: response_path}
    return Response(response)


@api_view(["GET"])
def escalation_api_view(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper over Escaltion classifier
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"escalation_cls_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating allocator specs from specs.yaml file
    escalation_specs = specs.get_escalation_specs()
    escalation_input = escalation_specs.get("input")
    input_file_name = escalation_specs.get("input_file_name")
    escalation_output = escalation_specs.get("output")
    output_file_name = escalation_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        escalation_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass to serve function
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    escalation_response_path = (
        f"{output_base_path}{escalation_output}/{output_file_name}"
    )
    response_path = (
        f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{escalation_response_path}"
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
        # Calls handle_failure function to update summarizer status to error
        handle_failure(
            file_obj=file_obj,
            request_id=request_id,
            model_name=constants.ESCALATION_CLS,
            response_path=escalation_response_path,
        )

        # Returns error response
        response = {constants.RESPONSE[constants.ESCALATION_CLS]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.ESCALATION_CLS],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(
        output_path=escalation_response_path, output=progress_response
    )

    # Initializing kwargs for background process and listener
    listener_dict = {
        "request_id": request_id,
        "scheduler": scheduler_obj_name,
        "model": constants.ESCALATION_CLS,
        "response_path": escalation_response_path,
    }
    param_dict = {
        "input_file": input_file,
        "scheduler": scheduler_obj_name,
        "serve_function": escalation_model,
        "response_path": escalation_response_path,
        "model": constants.MODEL_NAMES[constants.ESCALATION_CLS],
    }
    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.ESCALATION_CLS]: response_path}
    return Response(response)
