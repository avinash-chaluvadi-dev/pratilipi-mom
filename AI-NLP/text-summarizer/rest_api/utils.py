import datetime
import json
import logging as lg
import re
import traceback
from collections import OrderedDict
from copy import deepcopy
from functools import reduce

from django.conf import settings
from django.db.models import Count
from rest_framework.response import Response

from boiler_plate.utility import constants
from boiler_plate.utility.http import logger
from boiler_plate.utility.utils import S3Utils
from rest_api.models import File

logger = lg.getLogger("file")


def filter_label_from_list(label_dict: dict, filter_value: list):
    """Returns True if atleast one label is common
    between label_dict and filter_value"""

    if not bool(label_dict):
        return False
    if len(set(list(label_dict.keys())).intersection(filter_value)) > 0:
        return True
    else:
        return False


def filter_entity_from_list(entity_list: list, filter_value: list):
    """Returns True if atleast one entity type is
    common between entity_list and filter_value"""

    if not bool(entity_list[0][constants.MOM_ENTITY_TYPE]):
        return False
    if (
        len(set(entity_list[0][constants.MOM_ENTITY_TYPE]).intersection(filter_value))
        > 0
    ):
        return True
    else:
        return False


def detailed_view_filter(mom_response: dict, data_filter_payload: dict):
    logger.info(f"Data filter payload --> {data_filter_payload}")
    # Initializes empty list/dictionary to hold filtered response
    filtered_list = []
    filter_response = {}
    # Iterates over data_filter payload to filter corresponding transcripts from mom response
    for filter_key, filter_value in data_filter_payload.items():
        if bool(filter_value):
            empty_filter = False
            if filter_key == constants.FILTER_LABELS:
                logger.info(
                    f"Started filtering {filter_value} corresponding to {filter_key} from mom response file"
                )
                label_filter_response = [
                    transcript
                    for transcript in mom_response.get(constants.MOM_CONCATENATED_VIEW)
                    if filter_label_from_list(
                        label_dict=transcript.get(constants.MOM_LABEL),
                        filter_value=filter_value,
                    )
                ]
                # Appends manual_entries corresponding transcripts
                if bool(mom_response.get(constants.MOM_MANUAL_ENTRIES, {})):
                    for manual_label, manual_entries in mom_response.get(
                        constants.MOM_MANUAL_ENTRIES
                    ).items():
                        if manual_label in filter_value:
                            label_filter_response += manual_entries
                if len(label_filter_response) > 0:
                    logger.info(
                        f"Initializing empty dictionary to filter corresponding {filter_value}"
                    )
                    filter_response[filter_key] = {}
                    filter_response[filter_key] = label_filter_response
                    filtered_list.append(label_filter_response)

            elif filter_key == constants.FILTER_ENTITIES:
                logger.info(
                    f"Started filtering {filter_value} corresponding to {filter_key} from mom response file"
                )
                filter_value = [
                    constants.ENTITY_UI_MODEL_MAP.get(entity) for entity in filter_value
                ]
                entity_filter_response = [
                    transcript
                    for transcript in mom_response.get(constants.MOM_CONCATENATED_VIEW)
                    if filter_entity_from_list(
                        entity_list=transcript.get(constants.MOM_ENTITIES),
                        filter_value=filter_value,
                    )
                ]

                if len(entity_filter_response) > 0:
                    logger.info(
                        f"Initializing empty dictionary to filter corresponding {filter_value}"
                    )
                    filter_response[filter_key] = {}
                    filter_response[filter_key] = entity_filter_response
                    filtered_list.append(entity_filter_response)

            elif filter_key == constants.FILTER_SENTIMENTS:
                logger.info(
                    f"Started filtering {filter_value} corresponding to {filter_key} from mom response file"
                )
                sentiment_filter_response = [
                    transcript
                    for transcript in mom_response.get(constants.MOM_CONCATENATED_VIEW)
                    if filter_label_from_list(
                        label_dict=transcript.get(constants.MOM_SENTIMENT),
                        filter_value=filter_value,
                    )
                ]

                if len(sentiment_filter_response) > 0:
                    logger.info(
                        f"Initializing empty dictionary to filter corresponding {filter_value}"
                    )
                    filter_response[filter_key] = {}
                    filter_response[filter_key] = sentiment_filter_response
                    filtered_list.append(sentiment_filter_response)

            elif filter_key == constants.FILTER_PARTICIPANTS:
                logger.info(
                    f"Started filtering {filter_value} corresponding to {filter_key} from mom response file"
                )
                participant_filter_response = [
                    transcript
                    for transcript in mom_response.get(constants.MOM_CONCATENATED_VIEW)
                    if transcript.get(constants.MOM_SPEAKER_ID) in filter_value
                ]

                if len(participant_filter_response) > 0:
                    logger.info(
                        f"Initializing empty dictionary to filter corresponding {filter_value}"
                    )
                    filter_response[filter_key] = {}
                    filter_response[filter_key] = participant_filter_response
                    filtered_list.append(participant_filter_response)

        else:
            empty_filter = True
            logger.info(
                f"No filter specified for {filter_key}, skipping mom response filtering"
            )

    if len(filtered_list) > 0:
        filtered_list = reduce(lambda z, y: z + y, filtered_list)

        filter_response[constants.FILTER_FLATENNED_LIST] = list(
            {
                value[constants.MOM_SPEAKER_LABEL]: value for value in filtered_list
            }.values()
        )
    elif empty_filter:
        filter_response = deepcopy(mom_response)

    return filter_response


def dashboard_file_status_counts(file_object, data_dict) -> Response:
    try:
        status_dict = {
            status_dict.get("status"): status_dict.get("total_count")
            for status_dict in file_object.values("status")
            .annotate(total_count=Count("status"))
            .order_by()
        }
        if file_object.count() > 0:
            data_dict["dashboard_info"]["recordings_uploaded"] = file_object.count()
            data_dict["dashboard_info"]["mom_ready_for_review"] = status_dict.get(
                "Ready for Review", 0
            )
            data_dict["dashboard_info"]["mom_in_review"] = status_dict.get(
                "User Review In Progress", 0
            )
            data_dict["dashboard_info"]["mom_generated"] = status_dict.get(
                "Completed", 0
            )
        else:
            data_dict["dashboard_info"]["recordings_uploaded"] = 0
            data_dict["dashboard_info"]["mom_ready_for_review"] = 0
            data_dict["dashboard_info"]["mom_in_review"] = 0
            data_dict["dashboard_info"]["mom_generated"] = 0

    except Exception as e:
        logger.error(
            f"Something went wrong while processing participants window consolidate request...!\nstr{e}"
        )
        raise traceback.format_exc()


def parse_string_datetime(start_date, end_date):
    pattern = re.compile(r"(\d+)([/-])(\d+)([/-])(\d+)")
    start_date_match = pattern.match(start_date)
    end_date_match = pattern.match(end_date)
    seperator = start_date_match.group(2)
    if int(start_date_match.group(3)) <= 12:
        if len(start_date_match.group(5)) > 2:
            start_date = datetime.datetime.strptime(
                start_date, f"%d{seperator}%m{seperator}%Y"
            ).date()
            end_date = datetime.datetime.strptime(
                end_date, f"%d{seperator}%m{seperator}%Y"
            ).date() + datetime.timedelta(days=1)
        elif len(start_date_match.group(5)) == 2:
            start_date = datetime.datetime.strptime(
                start_date, f"%d{seperator}%m{seperator}%y"
            ).date()
            end_date = datetime.datetime.strptime(
                end_date, f"%d{seperator}%m{seperator}%y"
            ).date() + datetime.timedelta(days=1)

    else:
        raise Exception("Incorrect date format")

    return start_date, end_date


def get_label_items(mom_json_path, labels_list):
    response = []
    with open(mom_json_path, "r") as f:
        mom_json_obj = json.load(f)
    for data_point in mom_json_obj.get("concatenated_view"):
        if data_point.get("label"):
            print(data_point.get("chunk_id"))

            response.append(
                {
                    label: confidence_score
                    for label, confidence_score in data_point.get("label").items()
                    if label in labels_list
                }
            )
    response = list(filter(None, response))
    return labels_list


def dashboard_label_counts(consolidated_data, data_dict):
    data_dict["dashboard_info"]["action_count"] = (
        consolidated_data.values("label").filter(label__contains="Action").count()
    )
    data_dict["dashboard_info"]["appreciation_count"] = (
        consolidated_data.values("label").filter(label__contains="Appreciation").count()
    )
    data_dict["dashboard_info"]["escalation_count"] = (
        consolidated_data.values("label").filter(label__contains="Escalation").count()
    )


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


def dashboard_label_details(**kwargs):
    label_list = ["Action", "Escalation", "Appreciation"]
    data_dict = kwargs.get("data_dict")
    mom_object = kwargs.get("mom_object")
    consolidated_data = kwargs.get("consolidated_data")
    counts = {"action": 0, "appreciation": 0, "escalation": 0}

    for entry in mom_object:
        team_name = str(File.objects.get(masked_request_id=entry.request_id).team_name)
        date = str(File.objects.get(masked_request_id=entry.request_id).date)
        json_file = str(entry.mom_output_v1)
        json_obj = load_json(json_file)
        for label in label_list:
            if json_obj.get("mom_entries").get(label) is not None:
                if not data_dict.get("team_info").get(team_name.capitalize()):
                    data_dict["team_info"][team_name.capitalize()] = OrderedDict()
                if not (
                    data_dict.get("team_info")
                    .get(team_name.capitalize())
                    .get(str(label.capitalize()))
                ):
                    data_dict["team_info"][team_name.capitalize()][
                        str(label.capitalize())
                    ] = OrderedDict()
                if not (
                    data_dict.get("team_info")
                    .get(team_name.capitalize())
                    .get(str(label.capitalize()))
                    .get("transcripts")
                    and data_dict.get("team_info")
                    .get(team_name.capitalize())
                    .get(str(label.capitalize()))
                    .get("count")
                ):

                    data_dict["team_info"][team_name.capitalize()][
                        str(label.capitalize())
                    ]["transcripts"] = []

                    data_dict["team_info"][team_name.capitalize()][
                        str(label.capitalize())
                    ]["count"] = 0

                data_dict["team_info"][team_name.capitalize()][str(label.capitalize())][
                    "count"
                ] += len(json_obj.get("mom_entries").get(label))

                data_dict["team_info"][team_name.capitalize()][str(label.capitalize())][
                    "transcripts"
                ].extend(
                    [
                        {
                            "transcript": transcript_dict.get("summary"),
                            "owner": transcript_dict.get("assign_to"),
                            "date": transcript_dict.get("date"),
                        }
                        for transcript_dict in json_obj.get("mom_entries").get(label)
                    ]
                )
                counts[label.lower()] += len(json_obj.get("mom_entries").get(label))
        # print(data_dict)
    data_dict["dashboard_info"]["action_count"] = counts["action"]
    data_dict["dashboard_info"]["escalation_count"] = counts["escalation"]
    data_dict["dashboard_info"]["appreciation_count"] = counts["appreciation"]
