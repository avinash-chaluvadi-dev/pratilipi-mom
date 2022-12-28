import copy
import json
import logging
import os
import re
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


class S3Adapter:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

        # Initializes s3 client and resource to interact with S3 service
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

    def read_json(self, read_path: str):
        """
        To read the json data from S3 in form of bytes
        and converts into dict object
        """
        try:
            if constants.S3_URI in read_path:
                # Removes the S3 URI headers from read_path
                read_path = self.get_prefix_from_uri(path=read_path)

            # Get the particular object from read_path of corresponding S3 bucket
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=read_path)
            return json.loads(obj["Body"].read().decode("utf-8"))

        except exceptions.DataNotFoundError as error:
            logger.error("Specified prefix does not exist")
            raise RuntimeError(error)

    def put_data(self, store_path, content):
        """
        To put the data in the form of bytes into S3
        """
        try:
            if constants.S3_URI in store_path:
                # Removes the S3 URI headers from read_path
                store_path = self.get_prefix_from_uri(path=store_path)

            # Put the data into store_path of corresponding S3 bucket
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=store_path,
                Body=content,
            )

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def publish_message(self, topic_arn: str, subject: str, message: dict):
        """
        Function to publish message into SNS Topic
        """

        # Publishes message into DIARIZATION SNS topic
        response = self.sns_client.publish(
            TopicArn=topic_arn,
            Subject=subject,
            MessageStructure="json",
            Message=json.dumps({"default": json.dumps(message)}),
        )

        # Returns sns confirmation/failure response
        return response

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


class TranscribeUtils:
    """For intializing and delting the Transcribe JOb"""

    def __init__(self, s3_utils: S3Adapter, job_name: str, job_status: str):
        self.s3_utils = s3_utils
        self.job_name = job_name
        self.job_status = job_status

        # Initializes transcribe client to interact with AWS Transcribe
        self.transcribe_client = boto3.client("transcribe")

    def cleanup(self):
        """Deletes transcribe job if it is in completed or failed status"""

        # Removes failed transcription job
        logger.info(f"Found job in {self.job_status} status, deleting it")

        logger.info(f"Started removing the transcription job")
        self.transcribe_client.delete_transcription_job(
            TranscriptionJobName=self.job_name
        )
        logger.info(f"Transcription job - {self.job_name} got successfully removed")

    def handle_transcribe_failure(
        self,
        extension: str,
        file_name: str,
        base_path: str,
        media_path: str,
        diarization: int,
    ):
        """Deletes transcribe job if it is in completed or failed status.

        :raises RuntimeError which should caught at main level and returned a proper response to client
        """

        # Cleans the existing transcription job
        logger.info(
            f"Process failed at transcript generation step for wav file {file_name}"
        )

        # Get S2T store path for stroing error response
        s2t_store_path = get_store_path(base_path=base_path)
        logger.info(f"AWS Transcribe output path --> {s2t_store_path}")

        # Get Diarization store path for stroing error response
        diarization_store_path = get_diarization_store_path(base_path=base_path)
        logger.info(f"Diarization output path --> {diarization_store_path}")

        # To put the error response into S3 bucket
        (
            error_message,
            s2t_error_response,
            diarization_error_response,
        ) = create_error_response(
            extension=extension,
            media_path=media_path,
            diarization=diarization,
            s2t_store_path=s2t_store_path,
            diarization_store_path=diarization_store_path,
        )

        # Stores speech to text and diarization error responses into S3 Bucket
        self.s3_utils.put_data(
            store_path=s2t_store_path, content=json.dumps(s2t_error_response)
        )
        self.s3_utils.put_data(
            store_path=diarization_store_path,
            content=json.dumps(diarization_error_response),
        )

        # Publisges a message using publish_message function
        response = self.s3_utils.publish_message(
            message=error_message,
            subject=constants.ERROR_SUBJECT,
            topic_arn=CONFIG.get("DOWNSTREAM_TOPIC_ARN"),
        )
        logger.info(
            f"Speech to Text/Diarization error message confirmation  --> {response}"
        )

        # Cleans/Removes failed transcription job
        self.cleanup()


def clean_path(path: str) -> str:
    """Removes extra slashes and trailing slashes from input path"""

    # Generates pattern for removing extra slashes from path/prefix
    extra_slash_pattern = re.compile(r"[//]+")

    # Returns cleaned path/prefix by stripping the path further
    return re.sub(extra_slash_pattern, "/", path).strip("/")


def get_base_path(path: str = None):
    """Returns base_path from either constants module or S3 URI"""

    # Checks whether constants module has BASE_PATH variable or not
    if hasattr(constants, "BASE_PATH"):
        # Assigns base_path to constants BASE_PATH attribute
        base_path = constants.BASE_PATH

    else:
        if path:
            # Extracts base name from input S3 URI
            base_path = (
                path.replace(constants.S3_URI, "")
                .replace(os.path.basename(path), "")
                .split("/", 1)[1]
            )

        else:
            # Assigns base_path to empty string
            base_path = ""

    return base_path


def get_store_path(base_path: str) -> str:
    """This function returns store path which
    aws transcribe uses to store the output
    """

    # To get the speech to text constants
    s2t_specs = constants.SPEECH_TO_TEXT

    # Parses/Gets store path from S3 URI/constants module
    store_path = os.sep.join(
        [
            base_path,
            s2t_specs.get("output_dir"),
            s2t_specs.get("output_file"),
        ]
    )

    return clean_path(store_path)


def get_diarization_store_path(base_path) -> str:
    """This function returns store path which
    aws transcribe uses to store diarization output
    """

    # To get the speech to text constants
    diarization_specs = constants.DIARIZATION

    # Parses/Gets store path from S3 URI/constants module
    store_path = os.sep.join(
        [
            base_path,
            diarization_specs.get("output_dir"),
            diarization_specs.get("output_file"),
        ]
    )

    return clean_path(store_path)


def get_bucket_name(s3_uri: str = None) -> str:
    """Returns bucket_name from either constants module or S3 URI"""

    # Checks whether constants module has BUCKET_NAME variable or not
    if hasattr(constants, "BUCKET_NAME"):
        bucket_name = constants.BUCKET_NAME
    else:
        if s3_uri:
            # Extracts bucket name from S3 URI
            bucket_name, _ = parse_media_uri(file_uri=s3_uri).split("/", 1)
        else:
            raise RuntimeError(
                "No Bucket Name specified for storing the aws transcribe output"
            )
    return bucket_name


def parse_media_uri(file_uri: str) -> str:
    """parses the media uri to generate prefix"""
    return file_uri.replace(constants.S3_URI, "")


def parse_transcript_uri(file_uri: str) -> str:
    """parses the transcript uri to generate prefix"""
    return file_uri.replace("https://s3.us-east-1.amazonaws.com/", "")


def create_s2t_response(status, transcript: dict):
    """Creates Speech to text response from specs in constants module"""

    # To get the speech to text constants
    s2t_specs = constants.SPEECH_TO_TEXT

    # Creates and returns dictionary with respective values
    s2t_response = {
        constants.STATUS: status,
        s2t_specs.get("model_key"): s2t_specs.get("model_val"),
        constants.RESPONSE: [transcript],
    }

    return s2t_response


def create_s2t_success_message(
    status: str,
    extension: str = None,
    media_path: str = None,
    diarization: int = None,
    s2t_output_path: str = None,
):
    """
    Function to create message dict with keys
    and values from constant module/function params
    """

    # To get the speech to text constants
    s2t_specs = constants.SPEECH_TO_TEXT

    # Creates and returns dictionary with respective values
    s2t_message = {
        constants.STATUS: status,
        s2t_specs.get("model_key"): s2t_specs.get("model_val"),
    }

    if status == constants.SUCCESS:
        logger.info("Generating Speech to Text success message")
        s2t_message[constants.EXTENSION_KEY] = extension
        s2t_message[constants.DIARIZATION_KEY] = diarization
        s2t_message[constants.MEDIA] = media_path
        s2t_message[constants.S2T_OUTPUT] = s2t_output_path
        logger.info(
            f"Speech to Text success message generated successfully --> {s2t_message}"
        )

    return s2t_message


def get_diarization_extension(transcribe_job_response: dict):
    """Returns diarization and extension value by parsing
    AWS Transcribe Job response
    """

    # Parses diarization value from Transcribe job response
    diarization = (
        transcribe_job_response.get("TranscriptionJob")
        .get("Settings")
        .get("ShowSpeakerLabels")
    )
    diarization = int(diarization or 0)

    # Get MediaFormat value from transcribe job response
    extension = transcribe_job_response.get("TranscriptionJob").get("MediaFormat")

    return diarization, extension


def create_error_response(
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

    # Log statements for further monitoring
    logger.info("Generating Speech to Text and Diarization error message")

    # To get the speech to text and Diarization constants
    s2t_specs = constants.SPEECH_TO_TEXT
    diarization_specs = constants.DIARIZATION

    # Creates and returns dictionary with respective values
    s2t_error_response = {
        constants.STATUS: constants.ERROR,
        s2t_specs.get("model_key"): s2t_specs.get("model_val"),
        constants.RESPONSE: [],
    }

    diarization_error_response = {
        constants.STATUS: constants.ERROR,
        diarization_specs.get("model_key"): diarization_specs.get("model_val"),
        constants.RESPONSE: [],
    }

    logger.info(
        f"Speech to Text error message generated successfully --> {s2t_error_response}"
    )
    logger.info(
        f"Diarization error message generated successfully --> {diarization_error_response}"
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

    # Returns s2t and diarization error response/message
    return error_message, s2t_error_response, diarization_error_response


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

    logger.info(f"File Status update response --> {status_response}")
    logger.info(f"Successfully updated the status of {file_name} to {status}")
