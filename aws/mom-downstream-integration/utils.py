import json
import logging
import os
import re
import uuid
from datetime import datetime
from distutils.util import strtobool

import boto3
import constants
import requests
import urllib3
from botocore import exceptions
from dotenv import dotenv_values

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# To disable/supress unverified https warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Creates an OrderedDict of env variables from .env file
CONFIG = dict(dotenv_values(".env"))


def sync_get(url: str, params: dict):
    """Synchronous Http GET method for Summarizer API"""

    # Initializes an empty dictionary for response
    response_dict = {}

    # Fetches ssl verify value from environment variable
    ssl_verify = bool(strtobool(CONFIG.get("SSL_VERIFY")))

    # Invokes Summarizer Endpoint using python requests library
    try:
        response = requests.get(
            url=url,
            verify=ssl_verify,
            params=params,
            headers={
                constants.AUTHORIZATION_KEY: f"{constants.TOKEN_KEY} {constants.TOKEN_VALUE}"
            },
        ).json()
        return response

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error while requesting {url}")

    except requests.exceptions.TooManyRedirects:
        logger.error(f"too many redirects while requests=ing {url}")

    except Exception as e:
        logger.error(
            "Unable to get url {} due to {}.".format(
                url, getattr(e, "message", repr(e))
            )
        )


def sync_post(url: str, data: dict):
    """Synchronous Http GET method for Summarizer API"""

    # Initializes an empty dictionary for response
    response_dict = {}

    # Fetches ssl verify value from environment variable
    ssl_verify = bool(strtobool(CONFIG.get("SSL_VERIFY")))

    # Invokes Summarizer Endpoint using python requests library
    try:
        response = requests.post(url=url, verify=ssl_verify, data=data).json()
        return response

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error while requesting {url}")

    except requests.exceptions.TooManyRedirects:
        logger.error(f"too many redirects while requests=ing {url}")

    except Exception as e:
        logger.error(
            "Unable to get url {} due to {}.".format(
                url, getattr(e, "message", repr(e))
            )
        )


def get_file_prefix(path: str):
    """Returns base prefix by parsing input S3 URI"""

    return path.replace(constants.S3_URI, "").split("/", 1)[1]


def get_bucket_name(s3_uri: str = None) -> str:
    """Returns bucket_name from either constants module or S3 URI"""

    # Checks whether constants module has BUCKET_NAME variable or not
    if hasattr(constants, "BUCKET_NAME"):
        bucket_name = constants.BUCKET_NAME
    else:
        if s3_uri:
            # Extracting bucket name from S3 URI
            bucket_name, _ = s3_uri.replace(constants.S3_URI, "").split("/", 1)

        else:
            raise RuntimeError(
                "No Bucket Name specified for storing the aws transcribe output"
            )
    return bucket_name


def clean_path(path: str) -> str:
    """Removes extra slashes and trailing slashes from input path"""

    # Generates pattern for removing extra slashes from path/prefix
    extra_slash_pattern = re.compile(r"[//]+")

    # Returns cleaned path/prefix by stripping the path further
    return re.sub(extra_slash_pattern, "/", path).strip("/")


def get_masked_request_id(media_uri: str):
    """Returns masked request id from mysql DB"""

    # Fetches ssl verify value from environment variable
    summarizer_url = CONFIG.get("SUMMARIZER_URL")
    logger.info(f"MOM Summarizer base endpoint --> {summarizer_url}")
    # Creates upload endpoint from summarizer base url and constants module
    upload_url = f"{summarizer_url}/{constants.UPLOAD_SUFFIX}"
    logger.info(f"Upload Endpoint for filtering masked_request_id --> {upload_url}")

    # Get prefix from the media path url extracted from SNS topic message
    media_prefix = get_file_prefix(path=media_uri)
    logger.info(f"Media Prefix for filtering masked_request_id --> {media_prefix}")

    # Creates query params dict required for UPLOAD API
    params = {"file": media_prefix}

    # Invokes sync_get function which in turn invokes summarizer API
    response = sync_get(url=upload_url, params=params)
    logger.info(f"Filtered Response corresponding to {media_prefix} --> {response}")

    # Returns api_file object response corresponding to the media_uri
    return response


def get_state_machine_input(
    keyframes: int, wait_time: int, masked_request_id: str, backend_start_time: datetime
):
    """Creates input dictionary for state machine"""

    state_input_dict = {}

    state_input_dict[constants.KEYFRAMES] = keyframes
    state_input_dict[constants.REQUEST_ID_KEY] = masked_request_id
    state_input_dict[constants.WAIT_TIME] = CONFIG.get("WAIT_TIME")
    state_input_dict[constants.BACKEND_START_TIME] = backend_start_time.strftime(
        constants.BACKEND_TIME_FORMAT
    )

    # Returns Json string of all the requred params for state machine to execute
    return json.dumps(state_input_dict)


def get_state_machine_exec_name():
    """Returns a unique name for step functions to use for state machine"""

    # Gets base job name from constants module
    state_machine_exec_name = constants.STATE_MACHINE_EXECUTION_NAME

    # Generates uuid
    random_id = uuid.uuid4().hex

    # Generates current time stamp
    present_datetime = datetime.now().strftime("%H-%M-%d-%m-%Y")

    # Returns state machine execution name
    logger.info("Generating State Machine Execution name")
    return f"{state_machine_exec_name}_{present_datetime}_{random_id}"


def start_step_function_execution(state_input: str):
    """
    start_step_function_execution calls downstream state machine

    """

    # Initializes step-function client to interact with AWS state machine
    client = boto3.client("stepfunctions")

    # Get's unique execution name for state machine to use
    execution_name = get_state_machine_exec_name()
    logger.info(
        f"State Machine execution name generated successfully --> {execution_name}"
    )

    # Call start_execution method to invoke state machine
    try:
        response = client.start_execution(
            stateMachineArn=constants.STATE_MACHINE_ARN,
            name=execution_name,
            input=state_input,
        )
        return response

    except exceptions.ClientError as error:
        logger.error(error.response["Error"]["Message"])
        raise RuntimeError(error)


def file_status_update(status: str, file_name: str):
    """Updates the status of api_file table based on request_id/file_name"""

    summarizer_url = CONFIG.get("SUMMARIZER_URL")
    logger.info(f"MOM Summarizer base endpoint --> {summarizer_url}")
    # Creates upload endpoint from summarizer base url and constants module
    file_status_url = f"{summarizer_url}/{constants.STATUS_UPDATE_SUFFIX}"
    logger.info(
        f"Endpoint for updating file_status to ({status}) --> {file_status_url}"
    )

    if constants.S3_URI in file_name:
        bucket_name = get_bucket_name(s3_uri=file_name)
        file_name = clean_path(
            path=file_name.replace(constants.S3_URI, "").replace(bucket_name, "")
        )

    logger.info(f"Updating the status of {file_name} to {status} in '(api_file)' table")

    # Creates data dict for file-status update API(POST)
    data = {
        constants.STATUS: status,
        constants.FILE_NAME: file_name,
    }

    # Invokes sync_postfunction which in turn invokes file-status update API
    status_response = sync_post(url=file_status_url, data=data)

    # Log statements for further monitoring
    logger.info(f"File Status update response --> {status_response}")
    logger.info(f"Successfully updated the status of {file_name} to {status}")


def backend_start_time_update(backend_start_time: datetime, file_name: str):
    """Updates the backend_start_time of api_file table based on request_id/file_name"""

    summarizer_url = CONFIG.get("SUMMARIZER_URL")
    logger.info(f"MOM Summarizer base endpoint --> {summarizer_url}")
    # Creates upload endpoint from summarizer base url and constants module
    file_status_url = f"{summarizer_url}/{constants.STATUS_UPDATE_SUFFIX}"
    logger.info(
        f"Endpoint for updating backend_start_time to ({backend_start_time}) --> {file_status_url}"
    )

    if constants.S3_URI in file_name:
        bucket_name = get_bucket_name(s3_uri=file_name)
        file_name = clean_path(
            path=file_name.replace(constants.S3_URI, "").replace(bucket_name, "")
        )

    logger.info(
        f"Updating the backend_start_time of {file_name} to {backend_start_time} in '(api_file)' table"
    )

    # Creates data dict for file-status update API(POST)
    data = {
        constants.FILE_NAME: file_name,
        constants.BACKEND_START_TIME: backend_start_time,
    }

    # Invokes sync_postfunction which in turn invokes file-status update API
    status_response = sync_post(url=file_status_url, data=data)

    # Log statements for further monitoring
    logger.info(f"Backend start time update response --> {status_response}")
    logger.info(
        f"Successfully updated backend_start_time of {file_name} to {backend_start_time}"
    )


def run_keyframes(extension: str, diarization: int) -> int:
    """Function to check whether to invoke Keyframe MS from
    State Machine or not, returns bool value from either
    constants module or query params"""

    if CONFIG.get("KEYFRAMES"):
        # Returns the KEYFRAMES bool value from env file
        return CONFIG.get("KEYFRAMES")

    elif diarization and extension in constants.VIDEO_EXTENSIONS:
        # Returns the same value that is passed from query param
        logger.info(f"Keyframe Extractor/Classifier will be invoked from State Machine")
        return 1

    else:
        # Returns 0 when above cases fail
        logger.info(
            f"Keyframe Extractor/Classifier will not be invoked from State Machine"
        )
        return 0
