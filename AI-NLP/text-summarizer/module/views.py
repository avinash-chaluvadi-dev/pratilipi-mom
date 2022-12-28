import json
import logging as lg
import os
import shutil
import uuid
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from boiler_plate.utility import constants
from boiler_plate.utility.utils import (S3Utils, create_error_response,
                                        get_file_base_path,
                                        get_input_file_as_dict,
                                        write_to_output_location)
from module import specs
from module.Headliner import model_serve as headliner_model
from rest_api.models import FeedBackLoop, File, MeetingMetadata

from .feedback_adapter import feedback_cleaner
from .Summarizer import model_serve as summarizer_model
from .utils import (create_keyframes_output, delete_labels_detailed_view,
                    generate_mom, get_manually_added_labels, get_speaker_label,
                    handle_failure, insert_mom, labels_info,
                    models_output_exist, scheduler_config, update_mom,
                    update_mom_entries)

logger = lg.getLogger("file")


def background_run(**kwargs: dict):
    model = kwargs.get("model")
    scheduler = kwargs.get("scheduler")
    input_file = kwargs.get("input_file")
    response_path = kwargs.get("response_path")
    serve_function = kwargs.get("serve_function")

    # Calls ML serve/inference function
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
def text_summarizer_api_view(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper over Text Summarizer Model
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"summarizer_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating headline specs from specs.yaml file
    summarizer_specs = specs.get_summarizer_specs()
    summarizer_input = summarizer_specs.get("input")
    input_file_name = summarizer_specs.get("input_file_name")
    summarizer_output = summarizer_specs.get("output")
    output_file_name = summarizer_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        summarizer_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass to serve function
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    summarizer_response_path = (
        f"{output_base_path}{summarizer_output}/{output_file_name}"
    )
    response_path = f"{constants.S3_URI}{settings.AWS_STORAGE_BUCKET_NAME}/{summarizer_response_path}"

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
            response_path=summarizer_response_path,
            model_name=constants.MODEL_NAMES[constants.SUMMARIZER_MODEL],
        )

        # Returns error response
        response = {constants.RESPONSE[constants.SUMMARIZER_MODEL]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.SUMMARIZER_MODEL],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(
        output_path=summarizer_response_path, output=progress_response
    )

    # Initializing kwargs for background process and listener
    listener_dict = {
        "request_id": request_id,
        "scheduler": scheduler_obj_name,
        "response_path": summarizer_response_path,
        "model": constants.MODEL_NAMES[constants.SUMMARIZER_MODEL],
    }
    param_dict = {
        "input_file": input_file,
        "scheduler": scheduler_obj_name,
        "serve_function": summarizer_model,
        "response_path": summarizer_response_path,
        "model": constants.MODEL_NAMES[constants.SUMMARIZER_MODEL],
    }

    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.SUMMARIZER_MODEL]: response_path}
    return Response(response)


@api_view(["GET"])
def headliner_generation(request, *args, **kwargs):
    """
    API wrapper over Headliner Model
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Creates unique random id and present datetime string, which in turn be used as name
    random_id = uuid.uuid4().hex
    present_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%f")
    # Creates a scheduler object for background execution
    scheduler_obj_name = f"headliner_scheduler_{random_id}_{present_datetime}"

    # Get's scheduler_config object which get's used by BackgroundScheduler
    SCHEDULER_CONFIG = scheduler_config()
    globals()[scheduler_obj_name] = BackgroundScheduler(SCHEDULER_CONFIG)

    # Extrcating headline specs from specs.yaml file
    headliner_specs = specs.get_headliner_specs()
    headliner_input = headliner_specs.get("input")
    input_file_name = headliner_specs.get("input_file_name")
    headliner_output = headliner_specs.get("output")
    output_file_name = headliner_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        headliner_input,
        file_obj.get_file_name(),
    )

    # Loading input file as dict to pass to serve function
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    output_base_path = str(file_obj.get_file()).replace(file_obj.get_file_name(), "")
    headliner_response_path = f"{output_base_path}{headliner_output}/{output_file_name}"
    response_path = f"{constants.S3_URI}{settings.AWS_STORAGE_BUCKET_NAME}/{headliner_response_path}"

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
            response_path=headliner_response_path,
            model_name=constants.MODEL_NAMES[constants.HEADLINER_MODEL],
        )

        # Returns error response
        response = {constants.RESPONSE[constants.HEADLINER_MODEL]: response_path}
        return Response(response)

    # Writing inprogress response into S3 Buket
    progress_response = {
        constants.MODEL_KEY: constants.MODEL_NAMES[constants.HEADLINER_MODEL],
        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
    }
    write_to_output_location(
        output_path=headliner_response_path, output=progress_response
    )

    # Initializing kwargs for background process and listener
    listener_dict = {
        "request_id": request_id,
        "scheduler": scheduler_obj_name,
        "response_path": headliner_response_path,
        "model": constants.MODEL_NAMES[constants.HEADLINER_MODEL],
    }
    param_dict = {
        "input_file": input_file,
        "scheduler": scheduler_obj_name,
        "serve_function": headliner_model,
        "response_path": headliner_response_path,
        "model": constants.MODEL_NAMES[constants.HEADLINER_MODEL],
    }

    # Adding background_listener/run function to scheduler
    globals()[scheduler_obj_name].add_listener(
        partial(background_listener, kwargs=listener_dict)
    )
    globals()[scheduler_obj_name].add_job(background_run, kwargs=param_dict)
    globals()[scheduler_obj_name].start()

    response = {constants.RESPONSE[constants.HEADLINER_MODEL]: response_path}
    return Response(response)


class MinutesOfMeetingAPIView(APIView):
    def get(self, request, *args, **kwargs) -> Response:
        # Fetching input file location using the REQUEST_ID
        request_id = kwargs.get("request_id")

        keyframes = request.GET.get(constants.KEYFRAMES)
        # Creates framify, keyframe_cls output from speaker diarization output
        if keyframes and int(keyframes) == 0:
            create_keyframes_output(request_id=request_id)

        # Extrcating MOM specs from specs.yaml file
        mom_specs = specs.get_mom_specs()
        mom_input = mom_specs.get("input")
        mom_output = mom_specs.get("output")
        output_file_name = mom_specs.get("output_file_name")

        file_obj = get_object_or_404(File, masked_request_id=request_id)
        base_path = get_file_base_path(
            file_obj.get_file(),
            mom_input,
            file_obj.get_file_name(),
        )

        mom_output_prefix = f"{base_path}/{mom_output}/{output_file_name}"
        output_exist = models_output_exist(base_path=base_path)

        # Get's status from api_file table
        db_status = file_obj.status
        if db_status == constants.ERROR_DB or not output_exist:
            # Calls handle_failure function to update headliner status to error
            handle_failure(
                file_obj=file_obj,
                write_output=False,
                request_id=request_id,
                model_name=constants.MODEL_NAMES["mom"],
            )

            # Returns error response
            response = {}
            logger.error(
                f"Exception occured while serving mom -> Model output doen't exist or execution failed"
            )
            response[constants.DATA_KEY] = {
                constants.ERROR_KEY: "Exception occured while serving mom -> Model output doen't exist or execution failed"
            }
            response[constants.STATUS_KEY] = status.HTTP_500_INTERNAL_SERVER_ERROR
            return Response(**response)

        # Calls generate_mom func to concatenate all outputs
        mom = generate_mom(request_id, file_obj)
        insert_mom(file_obj.get_team_name(), request_id, mom.get("concatenated_view"))
        write_to_output_location(
            mom_output_prefix,
            mom,
        )

        # Updates mom_output prefix in api_mom table
        fb, _ = FeedBackLoop.objects.get_or_create(
            request_id=request_id,
        )
        fb.mom_output_v1 = mom_output_prefix
        fb.is_mom_exist = True
        fb.save()
        return Response(mom)

    def patch(self, request, *args, **kwargs) -> Response:
        # Creates s3_utils object to interact with S3 using boto3
        s3_utils = S3Utils()

        # Fetches request_id and corresponding file object from api_file table
        request_id = kwargs.get("request_id")
        file_obj = get_object_or_404(File, masked_request_id=request_id)

        # Extrcating mom specs from specs.yaml file
        mom_specs = specs.get_mom_specs()
        mom_input = mom_specs.get("input")
        mom_output = mom_specs.get("output")
        output_file_name = mom_specs.get("output_file_name")
        v1_output_file_name = f"mom_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json"

        base_path = get_file_base_path(
            file_obj.get_file(),
            mom_input,
            file_obj.get_file_name(),
        )
        # Creates prefixes to store cleaned json and replace old json
        mom_output_prefix = f"{base_path}/{mom_output}/{output_file_name}"
        mom_v1_output_prefix = f"{base_path}/{mom_output}/{v1_output_file_name}"

        output_exist = s3_utils.prefix_exist(file_prefix=mom_output_prefix)
        # Checks whether mom.json file exist or not using api_mom table
        if (
            not output_exist
            or not FeedBackLoop.objects.filter(request_id=request_id).exists()
        ):
            # Creates and returns error response
            response = {}
            error_message = f"MOM output doesn't exist, pratilipi process is {constants.IN_PROGRESS_STATUS}/{constants/FAILED_STATUS}"
            response[
                constants.STATUS_KEY
            ] = status.status.HTTP_500_INTERNAL_SERVER_ERROR
            response[constants.DATA_KEY] = {constants.ERROR_KEY: error_message}
            logger.error(
                f"MOM output doesn't exist, pratilipi process is {constants.IN_PROGRESS_STATUS}/{constants/FAILED_STATUS}"
            )
            return Response(**response)

        else:
            try:
                if constants.MOM_MANUAL_ADD in request.data.keys():
                    # Updates final MOM Json file with add labels
                    mom_dict = get_input_file_as_dict(
                        file_path=mom_output_prefix, output_exist=output_exist
                    )
                    # Get's speaker label to use for manual labels
                    speaker_label = get_speaker_label(mom_dict=mom_dict)
                    request.data.update(
                        {
                            constants.MOM_SPEAKER_LABEL: speaker_label,
                            constants.MOM_TRANSCRIPT: request.data.get(
                                constants.MOM_SUMMARY
                            ),
                            constants.MOM_LABEL: list(
                                request.data.get(constants.MOM_LABEL).keys()
                            )[0],
                        }
                    )
                    update_mom(mom_dict=mom_dict, label_payload=request.data)
                    mom = deepcopy(mom_dict)

                elif constants.MOM_MANUAL_DELETE in request.data.keys():
                    label = request.data.get(constants.MOM_LABEL)
                    mom_dict = get_input_file_as_dict(
                        file_path=mom_output_prefix, output_exist=output_exist
                    )
                    del mom_dict[constants.MOM_ENTRIES][label]
                    if label in mom_dict.get(constants.MOM_MANUAL_ENTRIES, {}).keys():
                        del mom_dict[constants.MOM_MANUAL_ENTRIES][label]
                    delete_labels_detailed_view(label=label, mom_dict=mom_dict)
                    mom = deepcopy(mom_dict)

                else:
                    # Loads mom json response from S3
                    mom_dict = get_input_file_as_dict(
                        file_path=mom_output_prefix, output_exist=output_exist
                    )
                    # Opens Annotation adapter Rules Json to clean transcripts
                    with open(constants.ANNOTATION_ADAPTOR_RULES_PATH) as rules:
                        rules_dict = json.load(rules)
                        logger.info(
                            f"Rules being used for feedback cleanup are --> {rules_dict}"
                        )
                    logger.info(
                        f"Started feedback cleanup activity for request {request_id}"
                    )

                    # Call feedback_cleaner func to get cleaned transcript
                    output = feedback_cleaner(request.data, rules_dict)
                    mom = deepcopy(request.data)
                    mom["concatenated_view"] = output
                    # Get's manually added labels from MOM page
                    manual_labels_dict = mom_dict.get(constants.MOM_MANUAL_ENTRIES)
                    # Get's unique elements from mom json response using concateated_view
                    labels_elements = labels_info(mom)
                    # Updates mom_entries with changes done in detailed view
                    update_mom_entries(
                        mom_template=mom,
                        labels_elements=labels_elements,
                        manual_labels_dict=manual_labels_dict,
                    )

                # Uploads the cleaned Json file into S3 Bucket
                write_to_output_location(mom_output_prefix, mom)

                # Replaces the old mom.json with new cleaned json
                write_to_output_location(
                    mom_v1_output_prefix,
                    mom,
                )

                # Creates success response dict with 200 status
                response = {
                    constants.STATUS_KEY: status.HTTP_200_OK,
                    constants.DATA_KEY: mom,
                }

                if FeedBackLoop.objects.filter(request_id=request_id).exists():
                    feebackloop_obj = FeedBackLoop.objects.get(request_id=request_id)
                    feebackloop_obj.mom_output_v1 = mom_v1_output_prefix
                    feebackloop_obj.save()

            except KeyError as key_error:
                response[constants.STATUS_KEY] = status.HTTP_400_BAD_REQUEST
                response[constants.DATA_KEY] = {
                    constants.ERROR_KEY: f"Missing Body/Key in request"
                }
                logger.error(
                    f"Key error occured while feedback cleanup --> {str(key_error)}"
                )

            except RuntimeError as runtime_error:
                response[constants.DATA_KEY] = {
                    constants.ERROR_KEY: "Something went wrong while processing request"
                }
                response[constants.STATUS_KEY] = status.HTTP_500_INTERNAL_SERVER_ERROR
                logger.error(
                    f"Something went wrong while processing request --> str{runtime_error}"
                )

            # Returns cleaned response
            return Response(**response)
