import copy
import json
import logging
import os
import re
import uuid
from datetime import datetime
from distutils.util import strtobool
from time import time

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


class S3Adapter:
    """S3 Adapter which holds all S3 related utility functions to
    perform read, write operations using python SDK(boto3)
    """

    def __init__(self, bucket_name: str = None):
        """Constructor to initialize the state of object from method params"""

        # Instantiates the state of object from the input params
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")

        # Initializes a sns client to interact with SNS service
        self.sns_client = boto3.client("sns")

    def get_prefix_from_uri(self, path: str) -> str:
        """
        To generate the prefix from URI
        """
        # Generates the absolute path from root bucket and cleans using clean_path function
        return clean_path(
            path.replace(constants.S3_URI, "").replace(self.bucket_name, "").strip()
        )

    def prefix_exist(self, file_prefix):
        """Function to check whether prefix exist of not"""

        try:
            # Creates bucket object from s3 resource
            bucket = self.s3_resource.Bucket(self.bucket_name)

            logger.info(f"Started checking whether {file_prefix} exist or not")

            # Extracts prefix from s3 URL
            if constants.S3_URI in file_prefix:
                file_prefix = self.get_prefix_from_uri(path=file_prefix)

            if any(
                [
                    obj.key == file_prefix
                    for obj in list(bucket.objects.filter(Prefix=file_prefix))
                ]
            ):
                return True, file_prefix
            else:
                return False, file_prefix

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def put_data(self, store_path, content, content_type=None):
        """
        To put the data in the form of bytes into S3
        """

        try:
            if constants.S3_URI in store_path:
                # Removes the S3 URI headers from read_path
                store_path = self.get_prefix_from_uri(path=store_path)

            if not content_type:
                # Put the data into store_path of corresponding S3 bucket
                self.s3_client.put_object(
                    Bucket=self.bucket_name, Key=store_path, Body=content
                )
            else:
                # Put the data into store_path of corresponding S3 bucket
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=store_path,
                    Body=content,
                    ContentType=content_type,
                )

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def publish_message(self, message: dict, topic_arn: str, subject: str):

        """
        Function to publish message into SNS Topic

        """
        try:
            logger.info(
                "Started publishing the message into {mom-downstream-integration} topic"
            )

            # Publishes message into DIARIZATION SNS topic
            response = self.sns_client.publish(
                TopicArn=topic_arn,
                Subject=subject,
                MessageStructure="json",
                Message=json.dumps({"default": json.dumps(message)}),
            )
            logger.info(
                "Message message published successfully into the {mom-downstream-integration} topic"
            )
            return response

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)


def timeit_logger(func):
    """Decorator for logging time taken for a function to be executed"""

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        logger.info(f"Function {func.__name__} executed in {(time()-t1):0.2f}s")
        return result

    return wrap_func


def do_diarization(config, diarization: int = None) -> int:
    """Returns bool value from either constants module or query params"""

    if diarization:

        # Returns the same value that is passed from query param
        return int(diarization)

    elif config.get("SPEAKER_DIARIZATION"):
        # Returns the bool value from SPEAKER_DIARIZATIOn env varible
        return config.get("SPEAKER_DIARIZATION")

    else:
        # Returns 0 when above cases fail
        return 0


def get_bucket_name(s3_uri: str = None) -> str:
    """Returns bucket_name from either constants module or S3 URI"""

    # Checks whether constants module has BUCKET_NAME variable or not
    if hasattr(constants, "BUCKET_NAME"):
        bucket_name = constants.BUCKET_NAME
    else:
        if s3_uri:
            # Extracts bucket name from S3 URI
            bucket_name, _ = s3_uri.replace(constants.S3_URI, "").split("/", 1)

        else:
            raise RuntimeError(
                "No Bucket Name specified for storing the aws transcribe output"
            )
    return bucket_name


def get_base_path(path: str = None):
    """Returns base_path from either constants module or S3 URI"""

    # Checks whether constants module has BASE_PATH variable or not
    if hasattr(constants, "BASE_PATH"):
        base_path = constants.BASE_PATH
    else:
        if path:
            # Extracts bucket name from S3 URI
            base_path = (
                path.replace(constants.S3_URI, "")
                .replace(os.path.basename(path), "")
                .split("/", 1)[1]
            )
        else:
            base_path = constants.S2T_DEFAULT_FILE
    return base_path


def get_transcribe_job_name():
    """Returns a unique name for aws transcribe to use"""

    # Gets base job name from constants module
    transcribe_job_name = constants.TRANSCRIBE_JOB_NAME

    # Generates uuid
    random_id = uuid.uuid4().hex

    # Generates current time stamp
    present_datetime = datetime.now().strftime("%H-%M-%d-%m-%Y")

    return f"{transcribe_job_name}_{present_datetime}_{random_id}"


def clean_path(path: str) -> str:
    """Removes extra slashes and trailing slashes from input path"""

    # pattern for removing extra slashes from string
    extra_slash_pattern = re.compile(r"[//]+")
    return re.sub(extra_slash_pattern, "/", path).strip("/")


def get_store_path_from_uri(file_uri: str):
    """
    Returns store_path from Input S3 URI if passed,
    otherwise it returns default file name from constants
    """
    # Get DIARIZATION_DIARIZATION_DEFAULT_FILE from get_segments_defualt_path func
    S2T_DEFAULT_FILE = get_default_store_path()

    # Returns DIARIZATION_DEFAULT_FILE as store path
    if not file_uri:
        return S2T_DEFAULT_FILE

    _, object_prefix = file_uri.replace(constants.S3_URI, "").split("/", 1)
    return object_prefix.replace(os.path.basename(object_prefix), S2T_DEFAULT_FILE)
    return object_prefix


def get_default_store_path():
    """
    This function returns the basename of speech to text
    output from either constants module or default static values
    """
    if not hasattr(constants, "S2T_DEFAULT_FILE"):
        # Assigns S2T_DEFAULT_FILE to a static value as it isn't there in constants
        S2T_DEFAULT_FILE = "speech_to_text.json"

    else:
        # sets DIARIZATION_DEFAULT_FILE value to constants.DIARIZATION_DEFAULT_FILE
        S2T_DEFAULT_FILE = constants.S2T_DEFAULT_FILE

    return S2T_DEFAULT_FILE


def get_store_path(bucket_name: str, file_uri: str, store_path: str = None):
    """This function returns store path which
    aws transcribe uses to store the output
    """

    if store_path:
        # User can pass desired output location
        return store_path

    # Lambda func for converting None to ""
    xstr = lambda s: s or ""

    logger.info("No file specified for storing the output")

    # base_path for storing the aws transcribe output
    base_path = get_base_path(path=file_uri)

    # Points base_path to None if file_uri is None
    if not file_uri:
        base_path = None

    # output directory and filename to store transcribe output
    if hasattr(constants, "SPEECH_TO_TEXT"):
        s2t_output_dir = constants.SPEECH_TO_TEXT.get("output_dir")
        s2t_output_file = constants.SPEECH_TO_TEXT.get("output_file")

    else:
        s2t_output_dir = None
        s2t_output_file = None

    if s2t_output_dir and s2t_output_file:
        # Generates store_path from params in constants module
        logger.info("Generating {store_path} from S2T specs in constants module")

        store_path = os.sep.join(
            [xstr(base_path), xstr(s2t_output_dir), xstr(s2t_output_file)]
        )

    elif s2t_output_dir or s2t_output_file:
        if s2t_output_dir:
            # Get DIARIZATION_DEFAULT_FILE value from get_default_store_path func
            S2T_DEFAULT_FILE = get_default_store_path()
            store_path = os.sep.join(
                [xstr(base_path), xstr(s2t_output_dir), S2T_DEFAULT_FILE]
            )

        elif s2t_output_file:
            # Assigns store_path to base_dir as s2t_output_dir is None
            store_path = os.sep.join(
                [xstr(base_path), xstr(s2t_output_dir), s2t_output_file]
            )

    else:
        # Generates store_path from S3 URI or from static values defined in get_store_path_from_uri
        logger.info("Generating {store_path} from Input file/S3 URI")
        store_path = get_store_path_from_uri(file_uri=file_uri)
        store_path = clean_path(path=store_path)

    # Creates an S3 URI of store path and removes, strips extra slashes, trailing slashes
    store_path = f"{constants.S3_URI}{bucket_name}/{clean_path(store_path)}"
    return store_path


def get_diarization_store_path(bucket_name: str, file_uri: str) -> str:
    """This function returns store path which
    aws transcribe uses to store diarization output
    """

    # To get the speech to text constants
    diarization_specs = constants.DIARIZATION

    # base_path for storing the aws transcribe output
    base_path = get_base_path(path=file_uri)

    # Parses/Gets store path from S3 URI/constants module
    store_path = os.sep.join(
        [
            base_path,
            diarization_specs.get("output_dir"),
            diarization_specs.get("output_file"),
        ]
    )

    # Creates an S3 URI of store path and removes, strips extra slashes, trailing slashes
    store_path = f"{constants.S3_URI}{bucket_name}/{clean_path(store_path)}"
    return store_path


def handle_failure(
    extension: str,
    media_path: str,
    diarization: int,
    s2t_store_path: str,
    diarization_store_path: str,
):
    """
    Function to create S2T/diarization error message dict with keys
    and values from constant module/function params
    """

    # Get's S2T and Diarization constants/specs
    s2t_specs = constants.SPEECH_TO_TEXT
    diarization_specs = constants.DIARIZATION

    # Creates S2T error response with respective values
    logger.info("Generating Speech to Text error response")
    s2t_error_response = {
        constants.STATUS: constants.ERROR,
        s2t_specs.get("model_key"): s2t_specs.get("model_val"),
        constants.RESPONSE: [],
    }
    # Log statements for further monitoring
    logger.info(
        f"Speech to Text error response generated successfully --> {s2t_error_response}"
    )

    # Creates Diarization error response with respective values
    logger.info("Generating Diarization error response")
    diarization_error_response = {
        constants.STATUS: constants.ERROR,
        diarization_specs.get("model_key"): diarization_specs.get("model_val"),
        constants.RESPONSE: [],
    }
    # Log statements for further monitoring
    logger.info(
        f"Diarization error response generated successfully --> {diarization_error_response}"
    )

    # Copies diarization response into a new object
    error_message = copy.deepcopy(diarization_error_response)
    # Remove constants.RESPONSE diraization_response to generate SNS message
    del error_message[constants.RESPONSE]

    # Adds extension and diarization to error message to be published into SNS
    error_message[constants.DIARIZATION_KEY] = diarization
    error_message[constants.EXTENSION_KEY] = extension

    # Adds s2t and diarization store path to error message to be published into SNS
    error_message[constants.MEDIA] = media_path
    error_message[constants.S2T_OUTPUT] = s2t_store_path
    error_message[constants.DIARIZATION_OUTPUT] = diarization_store_path
    # Log statement for further monitoring
    logger.info(
        f"Diarization/S2T error message generated successfully --> {error_message}"
    )

    # Returns s2t and diarization error response/message
    return error_message, s2t_error_response, diarization_error_response


def create_error_message(media_path: str = None):

    # Replaces media_path None value with empty string if not found
    if not media_path:
        media_path = ""

    # Get's S2T onstants/specs
    s2t_specs = constants.SPEECH_TO_TEXT

    # Creates S2T error response with respective values
    logger.info("Generating error message")
    error_message = {
        constants.STATUS: constants.ERROR,
        s2t_specs.get("model_key"): s2t_specs.get("model_val"),
        constants.MEDIA: media_path,
    }

    # Log statements for further monitoring
    logger.info(f"Error message generated successfully --> {error_message}")
    return error_message


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
