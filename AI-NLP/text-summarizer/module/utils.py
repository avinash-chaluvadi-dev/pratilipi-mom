import json
import logging as lg
import os
from pathlib import Path

from django.conf import settings
from django.shortcuts import get_object_or_404

from boiler_plate.utility import constants
from boiler_plate.utility.http import logger
from boiler_plate.utility.orm_bulkcnfg import SensemakerbulkDBMnger
from boiler_plate.utility.utils import (S3Utils, create_error_response,
                                        get_file_base_path,
                                        get_input_file_as_dict,
                                        write_to_output_location)
from module import specs
from rest_api.models import ConsolidateModelsData, FeedBackLoop, File, Team

logger = lg.getLogger("file")


def create_keyframes_output(request_id):
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Extrcating framify specs from specs.yaml file
    framify_specs = specs.get_framify_specs()
    framify_input = framify_specs.get("input")
    input_file_name = framify_specs.get("input_file_name")
    framify_output = framify_specs.get("output")
    framify_output_file_name = framify_specs.get("output_file_name")

    # Extracts keyframe specs from specs.yaml file
    keyframe_cls_specs = specs.get_keyframe_cls_specs()
    keyframe_cls_output = keyframe_cls_specs.get("output")
    keyframe_cls_output_file_name = keyframe_cls_specs.get("output_file_name")

    # Get's file_object corresponding to api_file table
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        framify_input,
        file_obj.get_file_name(),
    )
    input_file_path = str(Path(f"{base_path}/{input_file_name}"))
    framify_output_base_path = str(file_obj.get_file()).replace(
        file_obj.get_file_name(), ""
    )
    framify_response_path = (
        f"{framify_output_base_path}{framify_output}/{framify_output_file_name}"
    )

    keyframe_output_base_path = str(file_obj.get_file()).replace(
        file_obj.get_file_name(), ""
    )
    keyframe_cls_response_path = f"{keyframe_output_base_path}{keyframe_cls_output}/{keyframe_cls_output_file_name}"

    # Loading input file as dict to check status of S2T/Diarization
    output_exist = s3_utils.prefix_exist(file_prefix=input_file_path)
    # Loading input file as dict to pass to serve function
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

    else:
        write_to_output_location(output_path=framify_response_path, output=input_file)
        write_to_output_location(
            output_path=keyframe_cls_response_path, output=input_file
        )


def models_output_exist(base_path: str):
    """Returns True if all ml models output exist
    and status equals to SUCCESS, else returns False
    """
    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = S3Utils()

    # Get's model output paths and iterates to check status
    for model in constants.MODELS:
        output_prefix = f"{base_path}/{getattr(constants, model)}"
        # Loading input file as dict to check status of S2T/Diarization
        output_exist = s3_utils.prefix_exist(file_prefix=output_prefix)
        input_file = get_input_file_as_dict(
            file_path=output_prefix, output_exist=output_exist
        )
        if not output_exist:
            return False

        elif input_file.get(constants.STATUS_KEY) != constants.SUCCESS_KEY:
            return False
    return True


def scheduler_config():
    """Initializes apscheduler object using params from constants module"""

    # Get's executor and job_store from constants module
    executor = constants.EXECUTORS.get(constants.EXECUTOR_TYPE)
    job_store = constants.JOB_STORES.get(constants.JOB_STORE_TYPE)

    # Creates scheduler_config object using apscheduler params in constants module
    scheduler_config = {
        constants.JOB_STORE: {
            constants.JOB_STORE_KEY: constants.JOB_STORE_TYPE,
            constants.JOB_STORE_URL: job_store.get(constants.JOB_STORE_URL),
        },
        constants.EXECUTOR: {
            constants.EXECUTOR_KEY: executor.get(constants.EXECUTOR_VALUE),
            constants.MAX_WORKERS: executor.get(constants.MAX_WORKERS),
        },
        constants.COALESCE_KEY: constants.COALESCE_VALUE,
    }

    # Returns scheduler config object
    return scheduler_config


def get_speaker_label(mom_dict: dict):
    """Returns speaker_label from concatenated list"""
    concatenated_view = mom_dict.get(constants.MOM_CONCATENATED_VIEW)
    if not bool(mom_dict.get(constants.MOM_MANUAL_ENTRIES)):
        if len(concatenated_view) > 0:
            speaker_label = concatenated_view[-1].get(constants.MOM_SPEAKER_LABEL) + 1
        else:
            speaker_label = 0
    else:
        speaker_label = 0
        for label, manual_entry in mom_dict.get(constants.MOM_MANUAL_ENTRIES).items():
            manual_speaker_label = manual_entry[-1].get(constants.MOM_SPEAKER_LABEL) + 1
            if manual_speaker_label > speaker_label:
                speaker_label = manual_speaker_label
    return speaker_label


def create_manual_label_dict(label_payload: dict):
    """Creates a dictionary object corresponding to add label"""
    manual_label_dict = {}
    for key in [*constants.MOM_STRING, *constants.MOM_DICT, *constants.MOM_LIST]:
        if key in constants.MOM_STRING:
            manual_label_dict[key] = label_payload.get(key, "")
        elif key in constants.MOM_DICT:
            manual_label_dict[key] = label_payload.get(key, {})
        else:
            manual_label_dict[key] = label_payload.get(key, [])
    manual_label_dict[constants.MOM_ENTITIES] = constants.MOM_ENTITIES_DEFAULT
    return manual_label_dict


def update_mom(mom_dict: dict, label_payload: dict):
    """ "Updates MOM Json with manual entries"""
    mom_entry = create_manual_label_dict(label_payload=label_payload)
    # Initializes mom_entries/label key with empty dict if not present in mom json
    if not constants.MOM_ENTRIES in mom_dict.keys():
        mom_dict[constants.MOM_ENTRIES] = {}
        mom_dict[constants.MOM_ENTRIES][label_payload.get(constants.MOM_LABEL)] = []

    if not constants.MOM_MANUAL_ENTRIES in mom_dict.keys():
        mom_dict[constants.MOM_MANUAL_ENTRIES] = {}
        mom_dict[constants.MOM_MANUAL_ENTRIES][
            label_payload.get(constants.MOM_LABEL)
        ] = []

    # Initializes label key with empty dict if not present in mom json
    if (
        not label_payload.get(constants.MOM_LABEL)
        in mom_dict.get(constants.MOM_ENTRIES).keys()
    ):
        mom_dict[constants.MOM_ENTRIES][label_payload.get(constants.MOM_LABEL)] = []

    if (
        not label_payload.get(constants.MOM_LABEL)
        in mom_dict.get(constants.MOM_MANUAL_ENTRIES).keys()
    ):
        mom_dict[constants.MOM_MANUAL_ENTRIES][
            label_payload.get(constants.MOM_LABEL)
        ] = []

    # Appends manually added mom entry to mom json
    mom_dict[constants.MOM_ENTRIES][label_payload.get(constants.MOM_LABEL)].append(
        mom_entry
    )
    mom_dict[constants.MOM_MANUAL_ENTRIES][
        label_payload.get(constants.MOM_LABEL)
    ].append(mom_entry)


def get_manually_added_labels(mom_dict: dict):
    """Creates dictionary of all the manually added labels"""

    manual_labels_dict = {}
    for label in mom_dict.get(constants.MOM_ENTRIES):
        manual_labels_dict[label] = [
            transcript
            for transcript in mom_dict.get(constants.MOM_ENTRIES).get(label)
            if transcript.get("manual_add") == "true"
        ]
    return manual_labels_dict


def delete_labels_detailed_view(label: str, mom_dict: dict):
    for transcript in mom_dict.get(constants.MOM_CONCATENATED_VIEW):
        label_dict = transcript.get(constants.MOM_LABEL)
        bkp_label_dict = transcript.get(constants.MOM_BKP_LABEL)
        if label in label_dict.keys():
            del label_dict[label]


def update_mom_entries(mom_template, labels_elements, manual_labels_dict=None):
    mom_template[constants.MOM_ENTRIES] = {}
    for label in labels_elements:
        label_fetch = []
        for chunk in mom_template["concatenated_view"]:
            if chunk["label"]:
                for key in chunk["label"].keys():
                    if key == label:
                        label_name = {
                            "speaker_label": chunk["speaker_label"],
                            "start_time": chunk["start_time"],
                            "end_time": chunk["end_time"],
                            "speaker_id": chunk["speaker_id"],
                            "transcript": chunk["transcript"],
                            "summary": chunk.get("summary"),
                            "sentiment": chunk["sentiment"],
                            "entities": chunk["entities"],
                            "keyframe_extractor": chunk.get("keyframe_extractor"),
                            "keyframe_labels": chunk.get("keyframe_labels"),
                            "marker": chunk.get("marker"),
                            "label": chunk["label"],
                            "assign_to": chunk["assign_to"],
                            "date": chunk["date"],
                        }

                        label_fetch.append(label_name)
        if label:
            mom_template[constants.MOM_ENTRIES][label] = label_fetch

    if bool(manual_labels_dict):
        if not constants.MOM_ENTRIES in mom_template.keys():
            mom_template[constants.MOM_ENTRIES] = {}
        for label, manual_entries in manual_labels_dict.items():
            if label in mom_template.get(constants.MOM_ENTRIES, {}).keys():
                mom_template[constants.MOM_ENTRIES][label] += manual_entries
            else:
                mom_template[constants.MOM_ENTRIES][label] = manual_entries


# TODO: Reduce the complexity of this function and remove the hardcoded localhost URL
def generate_mom(request_id: str, file_obj: object) -> dict:

    mom_template = ""
    summarizer_json = {}
    labels_elements = []
    if FeedBackLoop.objects.filter(request_id=request_id).exists():
        feedbackobj = FeedBackLoop.objects.get(request_id=request_id)
        outputpath = feedbackobj.get_mom_output_v1()
        mom_template = load_json(outputpath.__str__())
    else:
        summarizer_json = merge_ouputs(request_id)
        labels_elements = labels_info(summarizer_json)
        file_path = str(file_obj.get_file())
        if settings.USE_S3:
            # Importing S3Utils for s3 helper functions
            s3_utils = S3Utils()
            file_path = s3_utils.get_absolute_path(file_prefix=file_path)
        else:
            file_path = file_path.replace(
                os.path.abspath(settings.MEDIA_ROOT),
                constants.LOCAL_HOST_URL,
            )
        mom_template = {
            "meeting_id": "",
            "request_id": request_id,
            "overview": summarizer_json["overview"],
            "mom_entries": {},
            "date": file_obj.date.__str__(),
            "file_size": file_obj.file_size,
            "file_name": file_obj.get_file_name(),
            "file": file_path,
        }
        mom_template["concatenated_view"] = summarizer_json["concatenated_view"]

        update_mom_entries(mom_template=mom_template, labels_elements=labels_elements)

    # consolidateDatamgr.done()
    return mom_template


def handle_failure(
    file_obj: object,
    request_id: str,
    model_name: str,
    response_path: str = None,
    write_output: bool = True,
):
    """Function to handle failure and to generate error response"""

    logger.info(
        f"Pratilipi Speech To Text/Diarization Execution for {request_id} is failed/In Progress"
    )
    # Creates error response dictionary using constants module
    logger.info(f"Creating {model_name.upper()} error response")
    error_response = create_error_response(
        status=constants.ERROR_KEY, model=constants.MODEL_NAMES[model_name]
    )
    logger.info(
        f"{model_name.upper()} error response created successfully --> {error_response}"
    )
    if write_output:
        write_to_output_location(output_path=response_path, output=error_response)

    # Updates DB status to Error
    if file_obj.status != constants.ERROR_DB:
        logger.info(f"Updating Database status of {request_id}")
        file_obj.status = constants.ERROR_DB
        file_obj.save()
        logger.info(
            f"Database status of {request_id} updated successfully to {constants.ERROR_DB}"
        )


def insert_mom(team_id: int, request_id: str, file_obj: list) -> str:
    consolidate_datamgr = SensemakerbulkDBMnger(chunk_size=500)
    cons_data = ConsolidateModelsData.objects.filter(file_id=request_id)
    if cons_data.exists():
        cons_data.delete()
    for i in range(len(file_obj)):
        consolidate_datamgr.add(
            ConsolidateModelsData(
                file_id=request_id,
                team_id=team_id,
                participant_id=str(file_obj[i].get("speaker_id")),
                chunk_id=str(file_obj[i].get("chunk_id")),
                audio_path=str(file_obj[i].get("audio_path")),
                video_path=str(file_obj[i].get("video_path")),
                marker=str(file_obj[i].get("marker")),
                label=str(file_obj[i].get("label")),
                start_time=str(file_obj[i].get("start_time")),
                end_time=str(file_obj[i].get("end_time")),
                events=None,
            )
        )
    consolidate_datamgr.done()
    return "success"


def labels_info(summarizer_json):
    labels_list = []
    for x in summarizer_json["concatenated_view"]:
        op = list(x["label"].keys())
        labels_list.extend(op)
    return labels_list


def load_json(path):
    if settings.USE_S3:
        s3_utils = S3Utils()
        return s3_utils.load_json(load_prefix=path)
    else:
        try:
            with open(path) as file:
                json_data = json.load(file)
            return json_data
        except FileNotFoundError:
            logger.error(f"file with path {path} not found")
            return {}


def merge_ouputs(request_id):

    # Extrcating keyframe specs from specs.yaml file
    integration_specs = specs.get_integration_specs()
    output_file_name = integration_specs.get("output_file_name")

    # Fetching input file location using the REQUEST_ID
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file(),
        None,
        file_obj.get_file_name(),
    )

    # Loading Speech To Text Response
    speech_to_text = load_json(f"{base_path}/{constants.SPEECH_TO_TEXT}")

    # Loading Classifier responses
    ner_output = load_json(f"{base_path}/{constants.NER}")
    label_classifier = load_json(f"{base_path}/{constants.LABEL_CLASSIFIER}")
    sentiment_classifier = load_json(f"{base_path}/{constants.SENTIMENT_CLASSIFIER}")

    # Loading Summarizer responses
    headliner = load_json(f"{base_path}/{constants.HEADLINER}")
    text_summarizer = load_json(f"{base_path}/{constants.SUMMARIZER}")

    # Loading Keyframe responses
    keyframe_extractor = load_json(f"{base_path}/{constants.KEYFRAME_EXTRACTOR}")
    keyframe_classifier = load_json(f"{base_path}/{constants.KEYFRAME_CLASSIFIER}")
    resp = summarizer(
        request_id,
        speech_to_text,
        text_summarizer,
        label_classifier,
        sentiment_classifier,
        ner_output,
        keyframe_classifier,
        keyframe_extractor,
        headliner,
    )
    merge_output_prefix = f"{base_path}/{output_file_name}"

    if resp["status"] == constants.SUCCESS_KEY:
        write_to_output_location(
            merge_output_prefix,
            resp,
        )
        return resp
    else:
        return {"status": "fail"}


def convert_milli_to_timestamp(milli):
    millis = int(milli)
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = (millis / (1000 * 60 * 60)) % 24
    return str("%d:%d:%d" % (hours, minutes, seconds))


def summarizer(
    request_id,
    speech_to_text,
    text_summarizer,
    label_classifier,
    sentiment_classifier,
    ner_output,
    keyframe_label_classifier,
    keyframe_extractor,
    headliner_generation,
):
    """summarizer start"""
    # media_root = settings.MEDIA_ROOT
    if (
        speech_to_text["status"] == constants.SUCCESS_KEY
        and text_summarizer["status"] == constants.SUCCESS_KEY
    ):
        text_summarizer_output = text_summarizer[constants.RESPONSE_KEY]
        for item in speech_to_text[constants.RESPONSE_KEY]:
            for item2 in text_summarizer_output:
                if item["speaker_label"] == item2["speaker_label"]:
                    item["summary"] = item2["summary"]
                    item["start_time"] = convert_milli_to_timestamp(
                        item["start_time"] * 1000
                    )
                    item["end_time"] = convert_milli_to_timestamp(
                        item["end_time"] * 1000
                    )
                    item["date"] = ""
                    item["assign_to"] = ""

    if (
        speech_to_text["status"] == constants.SUCCESS_KEY
        and label_classifier["status"] == constants.SUCCESS_KEY
    ):
        label_classifier = label_classifier[constants.RESPONSE_KEY]
        sentiment_classifier = sentiment_classifier[constants.RESPONSE_KEY]
        ner_output = ner_output[constants.RESPONSE_KEY]
        keyframe_label_classifier = keyframe_label_classifier.get(
            constants.RESPONSE_KEY, {}
        )
        keyframe_extractor = keyframe_extractor.get(constants.RESPONSE_KEY, {})
        overview = headliner_generation["overview"]
        headliner_generation = headliner_generation[constants.RESPONSE_KEY]
        for item in speech_to_text[constants.RESPONSE_KEY]:
            item["label"] = {}
            for item2 in label_classifier:
                if (
                    item["speaker_label"] == item2["speaker_label"]
                    and item2.get("label")
                    and item2.get("label") != "Others"
                ):
                    item["label"] = {
                        item2.get("label"): item2.get("confidence_score"),
                    }
            for item3 in sentiment_classifier:
                if (
                    item["speaker_label"] == item3["speaker_label"]
                    and item3["classifier_output"]
                ):
                    item["sentiment"] = {
                        item3["classifier_output"]: item3["confidence_score"],
                    }
            for item5 in ner_output:
                if item["speaker_label"] == item5["speaker_label"]:
                    item["entities"] = [
                        item5["entities"],
                        item5["confidence_score"],
                    ]
            for item7 in keyframe_label_classifier:
                if item["speaker_label"] == item7.get("speaker_label"):
                    if item7.get("keyframes") and item7.get("keyframe_labels"):
                        item["keyframe_labels"] = [
                            item7.get("keyframes"),
                            item7.get("keyframe_labels"),
                            item7.get("confidence_score"),
                        ]
                    else:
                        item["keyframe_labels"] = []
                    break
            for item8 in keyframe_extractor:
                if item["speaker_label"] == item8["speaker_label"]:
                    if item8.get("keyframes"):
                        item["keyframe_extractor"] = [
                            item8["keyframes"],
                            item8["confidence_score"],
                        ]
                    else:
                        item["keyframe_extractor"] = []
            for item10 in headliner_generation:
                if item["speaker_label"] == item10["speaker_label"]:
                    item["headliner_generation"] = [
                        item10.get("classifier_output"),
                        item10.get("confidence_score"),
                    ]
            speech_to_text["request_id"] = request_id
            speech_to_text["overview"] = overview
            speech_to_text["model"] = "summarizer"
            speech_to_text["concatenated_view"] = speech_to_text["response"]
    resp = speech_to_text
    resp["concatenated_view"] = resp.pop("response")
    """ summarizer end"""
    return resp
