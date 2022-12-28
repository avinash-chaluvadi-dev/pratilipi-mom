import copy
import json
import logging
import os
import re
from distutils.util import strtobool
from string import punctuation

import boto3
import constants
import requests
import urllib3
from botocore import exceptions
from botocore.client import Config
from dotenv import dotenv_values
from text_processing import TextProcessing

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

    def __init__(self, bucket_name: str):
        """Constructor to initialize the state of object from method params"""

        # Instantiates the state of object from the input params
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3", config=Config(signature_version="s3v4"))

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

    def upload_file(self, temp_store_path, s3_store_path):
        """
        To upload a file the data in the form of bytes/string into S3
        """
        try:
            if constants.S3_URI in s3_store_path:
                # Removes the S3 URI headers from read_path
                s3_store_path = self.get_prefix_from_uri(path=s3_store_path)
            self.s3_resource.meta.client.upload_file(
                temp_store_path,
                self.bucket_name,
                s3_store_path,
                ExtraArgs={"ContentType": constants.VIDEO_CONTENT_TYPE},
            )

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def generate_presigned_uri(self, path, uri_timeout):
        """
        Generates the presigned URI/temporary URI of
        corresponding file object in S3 Bucket
        """
        try:
            # Let's add some log statements for further monitoring
            logger.info(f"Generating presigned URI of audio file --> {path}")

            if constants.S3_URI in path:
                # Removes the S3 URI headers from read_path
                path = self.get_prefix_from_uri(path=path)

            # Generates source signed uri/public uri which expires after the specified expiry time
            s3_source_signed_uri = self.s3_resource.meta.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": path},
                ExpiresIn=uri_timeout,
            )

            logger.info(
                f"Presigned URI generated successfully --> {s3_source_signed_uri}"
            )

            return s3_source_signed_uri

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def publish_message(self, diarization_message: dict, topic_arn: str, subject: str):

        """
        Function to publish message into SNS Topic

        """
        try:
            logger.info(
                "Started publishing the Diarization message into {mom-downstream-integration} topic"
            )

            # Publishes message into DIARIZATION SNS topic
            response = self.sns_client.publish(
                TopicArn=topic_arn,
                Subject=subject,
                MessageStructure="json",
                Message=json.dumps({"default": json.dumps(diarization_message)}),
            )
            logger.info(
                "Diarization message published successfully into the {mom-downstream-integration} topic"
            )
            return response

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)


def create_response(status: str, model_name: str, response: list):

    # To get the speech to text constants
    diarization_specs = constants.DIARIZATION

    logging.info(f"Generating {model_name} {status} response for MOM")
    # Creates and returns dictionary with respective values
    final_response = {
        constants.STATUS: status,
        diarization_specs.get("model_key"): model_name,
    }

    if status == constants.SUCCESS:
        final_response[constants.RESPONSE] = response

    else:
        final_response[constants.RESPONSE] = []

    logging.info(f"{model_name} {status} response generate successfully")

    return final_response


def create_success_message(
    extension: str,
    media_path: str,
    diarization: int,
    s2t_output_path: str,
    diarization_output_path: str,
):
    """
    Function to create diarization message with keys
    and values from constant module/function params
    """

    # To get the diarization constants
    diarization_specs = constants.DIARIZATION

    # Creates and returns dictionary with respective values
    diarization_message = {
        constants.STATUS: constants.SUCCESS,
        diarization_specs.get("model_key"): diarization_specs.get("model_val"),
    }

    logger.info("Generating Diarization Success message")

    # Adds extension and diarization values to SNS success message
    diarization_message[constants.EXTENSION_KEY] = extension
    diarization_message[constants.DIARIZATION_KEY] = diarization

    # Adds media path, s2t and diarization output paths to SNS success message
    diarization_message[constants.MEDIA] = media_path
    diarization_message[constants.S2T_OUTPUT] = s2t_output_path
    diarization_message[constants.DIARIZATION_OUTPUT] = diarization_output_path

    logger.info(
        f"Diarization success message generated successfully --> {diarization_message}"
    )

    # Returns diarization success
    return diarization_message


def handle_failure(
    extension: str,
    media_path: str,
    diarization: int,
    s2t_output_path: str,
    downstream_topic_arn: str,
    diarization_output_path: str,
):

    """Creates error response which will be stored in diarization store path
    and error message which will get published into Downstream SNS Topic
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

    # Copies diarization response into a new object to generate error message
    logger.info("Generating Diarization/S2T error message")
    error_message = copy.deepcopy(diarization_error_response)
    # Remove constants.RESPONSE diraization_response to generate SNS message
    del error_message[constants.RESPONSE]

    # Adds extension and diarization to error message to be published into SNS
    error_message[constants.EXTENSION_KEY] = extension
    error_message[constants.DIARIZATION_KEY] = diarization

    # Adds diarization store path to error message to be published into SNS
    error_message[constants.MEDIA] = media_path
    error_message[constants.S2T_OUTPUT] = s2t_output_path
    error_message[constants.DIARIZATION_OUTPUT] = diarization_output_path

    logger.info(
        f"Diarization/S2T error message generated successfully --> {error_message}"
    )

    # Returns error response/message
    return error_message, s2t_error_response, diarization_error_response


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


def get_segments_path_from_uri(file_uri: str):
    """
    Returns segments_path from Input S3 URI if passed,
    otherwise it returns default directory names from constants
    """

    # Get DIARIZATION_DEFAULT_AUDIO/VIDEO from get_segments_defualt_path func
    DIARIZATION_DEFAULT_AUDIO, DIARIZATION_DEFAULT_VIDEO = get_segments_defualt_path()

    # Returns DIARIZATION_DEFAULT_AUDIO/VIDEO as video/audio segments store path
    if not file_uri:
        return DIARIZATION_DEFAULT_AUDIO, DIARIZATION_DEFAULT_VIDEO

    # parses the file uri and replaces basename with DIARIZATION_DEFAULT_AUDIO/VIDEO names
    _, object_prefix = file_uri.replace(constants.S3_URI, "").split("/", 1)
    audio_path = object_prefix.replace(
        os.path.basename(object_prefix), DIARIZATION_DEFAULT_AUDIO
    )
    video_path = object_prefix.replace(
        os.path.basename(object_prefix), DIARIZATION_DEFAULT_VIDEO
    )
    return audio_path, video_path


def get_segments_defualt_path():
    # sets DIARIZATION_DEFAULT_FILE value if it doesn't exist in constants module
    if not hasattr(constants, "DIARIZATION_DEFAULT_AUDIO") and not hasattr(
        constants, "DIARIZATION_DEFAULT_VIDEO"
    ):
        DIARIZATION_DEFAULT_AUDIO = "audio_segments"
        DIARIZATION_DEFAULT_VIDEO = "video_segments"

    # sets DIARIZATION_DEFAULT_AUDIO to audio_segments
    elif not hasattr(constants, "DIARIZATION_DEFAULT_AUDIO"):
        DIARIZATION_DEFAULT_AUDIO = "audio_segments"
        DIARIZATION_DEFAULT_VIDEO = constants.DIARIZATION_DEFAULT_VIDEO

    # sets DIARIZATION_DEFAULT_VIDEO to video_segments
    elif not hasattr(constants, "DIARIZATION_DEFAULT_VIDEO"):
        DIARIZATION_DEFAULT_AUDIO = constants.DIARIZATION_DEFAULT_AUDIO
        DIARIZATION_DEFAULT_VIDEO = "video_segments"

    # sets DIARIZATION_DEFAULT_AUDIO/VIDEO value to constants.DIARIZATION_DEFAULT_AUDIO/VIDEO
    else:
        DIARIZATION_DEFAULT_AUDIO = constants.DIARIZATION_DEFAULT_AUDIO
        DIARIZATION_DEFAULT_VIDEO = constants.DIARIZATION_DEFAULT_VIDEO
    return DIARIZATION_DEFAULT_AUDIO, DIARIZATION_DEFAULT_VIDEO


def get_segments_path(file_uri: str, bucket_name: str):
    """This function returns audio path to which
    diarization stores the audio segments
    """
    # Checks and returns AUDIO_PATH and VIDEO_PATH from constants module
    if hasattr(constants, "AUDIO_PATH") and hasattr(constants, "VIDEO_PATH"):
        # Returns audio and video paths from constants module
        return constants.AUDIO_PATH, constants.VIDEO_PATH

    # Lambda func for converting None to ""
    xstr = lambda s: s or ""

    logger.info("No folder specified for storing the audio/video diarized segments")

    # base_path for storing the aws transcribe output
    base_path = get_base_path(path=file_uri)

    # Checks for output directory to store diarization segments
    if hasattr(constants, "DIARIZATION"):
        audio_segments = constants.DIARIZATION.get("audio_segments")
        video_segments = constants.DIARIZATION.get("video_segments")
        diarization_output_dir = constants.DIARIZATION.get("segment_dir")
    else:
        audio_segments = None
        video_segments = None
        diarization_output_dir = None

    if diarization_output_dir and audio_segments and video_segments:

        # Generates store_path from DIARIZATION params in constants module
        logger.info(
            "Generating {audio_segments} and {video_segments} path from Diarization specs in constants module"
        )
        audio_path = os.sep.join(
            [xstr(base_path), xstr(diarization_output_dir), xstr(audio_segments)]
        )
        video_path = os.sep.join(
            [xstr(base_path), xstr(diarization_output_dir), xstr(video_segments)]
        )

    elif diarization_output_dir or audio_segments or video_segments:

        # Generates audio/video paths from constants module attributes
        if diarization_output_dir:
            """
            Get DIARIZATION_DEFAULT_AUDIO/VIDEO and append those,
            to diarization_output_dir when audio/video paths are not defined"""
            # Get DIARIZATION_DEFAULT_AUDIO/VIDEO from get_segments_defualt_path func
            (
                DIARIZATION_DEFAULT_AUDIO,
                DIARIZATION_DEFAULT_VIDEO,
            ) = get_segments_defualt_path()

            if audio_segments:
                # Assigns DIARIZATION_DEFAULT_AUDIO to audio_segments path
                DIARIZATION_DEFAULT_AUDIO = audio_segments

            elif video_segments:
                # Assigns DIARIZATION_DEFAULT_VIDEO to video_segments path
                DIARIZATION_DEFAULT_VIDEO = video_segments

            # Generates full absolute path from root(bucket) to store audio/video segments
            audio_path = os.sep.join(
                [
                    xstr(base_path),
                    xstr(diarization_output_dir),
                    xstr(DIARIZATION_DEFAULT_AUDIO),
                ]
            )
            video_path = os.sep.join(
                [
                    xstr(base_path),
                    xstr(diarization_output_dir),
                    xstr(DIARIZATION_DEFAULT_VIDEO),
                ]
            )

        elif audio_segments or video_segments:
            if audio_segments:
                # Assigns DIARIZATION_DEFAULT_AUDIO to audio_segments path
                audio_path = os.sep.join(
                    [xstr(base_path), xstr(diarization_output_dir), audio_segments]
                )
                if not hasattr(constants, "DIARIZATION_DEFAULT_VIDEO"):
                    # Assigns DIARIZATION_DEFAULT_VIDEO to "video_segments" as it is not present in constants
                    DIARIZATION_DEFAULT_VIDEO = "video_segments"

                else:
                    # Assigns DIARIZATION_DEFAULT_VIDEO to DIARIZATION_DEFAULT_VIDEO
                    DIARIZATION_DEFAULT_VIDEO = constants.DIARIZATION_DEFAULT_VIDEO
                video_path = os.sep.join(
                    [
                        xstr(base_path),
                        xstr(diarization_output_dir),
                        DIARIZATION_DEFAULT_VIDEO,
                    ]
                )

            elif video_segments:
                if not hasattr(constants, "DIARIZATION_DEFAULT_AUDIO"):
                    # Assigns DIARIZATION_DEFAULT_AUDIO to "audio_segments" as it is not present in constants
                    DIARIZATION_DEFAULT_AUDIO = "audio_segments"

                else:
                    # Assigns DIARIZATION_DEFAULT_AUDIO to DIARIZATION_DEFAULT_AUDIO
                    DIARIZATION_DEFAULT_AUDIO = constants.DIARIZATION_DEFAULT_AUDIO
                audio_path = os.sep.join(
                    [
                        xstr(base_path),
                        xstr(diarization_output_dir),
                        DIARIZATION_DEFAULT_AUDIO,
                    ]
                )
                video_path = os.sep.join(
                    [xstr(base_path), xstr(diarization_output_dir), video_segments]
                )

    else:
        # Get audio_path and video_path from URI as none of the variables are present in constants module
        audio_path, video_path = get_segments_path_from_uri(file_uri=file_uri)

    # Creates an S3 URI of video/audio segments store path and removes, strips extra slashes, trailing slashes
    audio_path = f"{constants.S3_URI}{bucket_name}/{clean_path(audio_path)}"
    video_path = f"{constants.S3_URI}{bucket_name}/{clean_path(video_path)}"

    return audio_path, video_path


def get_segments_chunk(
    start_time: float, end_time: float, audio_path: str, video_path: str, extension: str
):
    """
    This function generates dynamic audio/video path
    for each segment of aws transcribe response using
    constants and base paths from settings
    """

    # Let's add some log statements for further monitoring
    logger.info(f"Generation of audio segment prefix has been started")
    logger.info(
        f"Audio segments will be saved into audio segment prefix --> {audio_path}"
    )

    # Generates audio basename which will be used by diarize function to store segments
    audio_chunk = (
        f"{constants.AUDIO_CHUNK}_{start_time}_{end_time}.{constants.AUDIO_EXTENSION}"
    )

    # Generates full absolute path for storing audio segments
    audio_path = os.sep.join([audio_path, audio_chunk])

    if extension in constants.VIDEO_EXTENSIONS:
        logger.info(
            f"Video segments will be saved into audio segment prefix --> {video_path}"
        )

        # Generates audio basename which will be used by diarize function to store segments
        video_chunk = f"{constants.VIDEO_CHUNK}_{start_time}_{end_time}.{constants.VIDEO_EXTENSION}"

        # Generates full absolute path for storing audio segments
        video_path = os.sep.join([video_path, video_chunk])

    else:
        logger.info(
            f" Extension type --> {extension}, skipping the generation of video segments path"
        )

        # Generates full absolute path for storing audio and video segments
        video_path = ""

    return audio_path, video_path


def get_default_store_path():
    """
    This function returns the basename of diarization output
    from either constants module or default static values
    """
    if not hasattr(constants, "DIARIZATION_DEFAULT_FILE"):
        # Assigns DIARIZATION_DEFAULT_FILE to a static value as it isn't there in constants
        DIARIZATION_DEFAULT_FILE = "speaker_diarization.json"

    else:
        # sets DIARIZATION_DEFAULT_FILE value to constants.DIARIZATION_DEFAULT_FILE
        DIARIZATION_DEFAULT_FILE = constants.DIARIZATION_DEFAULT_FILE

    return DIARIZATION_DEFAULT_FILE


def get_store_path_from_uri(file_uri: str):
    """
    Returns store_path from Input S3 URI if passed,
    otherwise it returns default file name from constants
    """
    # Get DIARIZATION_DIARIZATION_DEFAULT_FILE from get_segments_defualt_path func
    DIARIZATION_DEFAULT_FILE = get_default_store_path()

    # Returns DIARIZATION_DEFAULT_FILE as store path
    if not file_uri:
        return DIARIZATION_DEFAULT_FILE

    # parses the file uri and replaces basename with default filename
    _, object_prefix = file_uri.replace(constants.S3_URI, "").split("/", 1)
    store_path = object_prefix.replace(
        os.path.basename(object_prefix), DIARIZATION_DEFAULT_FILE
    )
    return store_path


def get_store_path(file_uri: str, bucket_name: str):
    """This function returns store path to which
    diarization stores the output
    """

    if hasattr(constants, "STORE_PATH"):
        # User can pass desired output location in constants module to store output
        return constants.STORE_PATH

    # Lambda func for converting None to ""
    xstr = lambda s: s or ""

    logger.info("No folder specified for storing the diarization output")

    # base_path for storing the diarization output
    base_path = get_base_path(path=file_uri)

    # Get output directory and filename to diarization output
    if hasattr(constants, "DIARIZATION"):
        # Takes and assigns store_path values from constants module
        diarization_output_dir = constants.DIARIZATION.get("output_dir")
        diarization_output_file = constants.DIARIZATION.get("output_file")

    else:
        # Assigns output dir/file values to None as they aren't available in constants
        diarization_output_dir = None
        diarization_output_file = None

    if diarization_output_dir and diarization_output_file:
        # Generates store_path from params in constants module
        logger.info(
            "Generating {store_path} from Diarization specs in constants module"
        )
        store_path = os.sep.join(
            [
                xstr(base_path),
                xstr(diarization_output_dir),
                xstr(diarization_output_file),
            ]
        )

    elif diarization_output_dir or diarization_output_file:
        if diarization_output_dir:
            # Get DIARIZATION_DEFAULT_FILE value from get_default_store_path func
            DIARIZATION_DEFAULT_FILE = get_default_store_path()
            store_path = os.sep.join(
                [
                    xstr(base_path),
                    xstr(diarization_output_dir),
                    DIARIZATION_DEFAULT_FILE,
                ]
            )

        elif diarization_output_file:
            # Assigns store_path to base_dir as diarization_output_dir is None
            store_path = os.sep.join(
                [xstr(base_path), xstr(diarization_output_dir), diarization_output_file]
            )
            store_path = clean_path(path=store_path)

    else:
        # Generates store_path from S3 URI or from static values defined in get_store_path_from_uri
        logger.info("Generating {store_path} from Input file/S3 URI")
        store_path = get_store_path_from_uri(file_uri=file_uri)

    # Creates an S3 URI of store path and removes, strips extra slashes, trailing slashes
    store_path = f"{constants.S3_URI}{bucket_name}/{clean_path(store_path)}"

    return store_path


def initilaize_segment_dict(
    speaker_id,
    speaker_label,
    start_time,
    end_time,
    audio_path,
    video_path,
    transcript,
    extension,
):
    """
    This function initializes empty dictionary
    for each item in transcribe response

    """

    # Creates a dictionary for speaker diarization segment
    segment_dict = {
        "manual_add": json.dumps(False),
        "speaker_id": speaker_id,
        "speaker_label": speaker_label,
        "start_time": start_time,
        "end_time": end_time,
        "audio_path": audio_path,
        "video_path": video_path,
        "transcript": transcript,
    }
    if extension not in constants.VIDEO_EXTENSION:
        segment_dict[
            constants.KEYFRAME_EXTRACTOR
        ] = constants.KEYFRAME_EXTRACTOR_DEFAULT
        segment_dict[
            constants.KEYFRAME_CLASSIFIER
        ] = constants.KEYFRAME_CLASSIFIER_DEFAULT

    return segment_dict


def get_segment_transcript(
    start_index: int, start_time: float, end_time: float, items_list: list
):
    """
    This function iterates over item list and generates
    transcript based on start and end time
    """

    count = 0
    total_items = len(items_list)
    words_list = []
    logger.info(f"Transcript Generation has been started for filename")
    for index, item in enumerate(items_list):
        if (
            item.get("start_time")
            and float(item.get("start_time")) >= start_time
            and float(item.get("end_time")) <= end_time
        ):
            count += 1
            words_list.append(item.get("alternatives")[0]["content"])
        elif not item.get("start_time") or not item.get("end_time"):
            if index == total_items - 1:
                count += 1
                words_list.append(item.get("alternatives")[0]["content"])
            elif float(items_list[index + 1].get("end_time")) <= end_time:
                count += 1
                words_list.append(item.get("alternatives")[0]["content"])
            else:
                words_list.append(item.get("alternatives")[0]["content"])
                break

    logger.info(f"Transcript Generation completed successfully")

    # Generates transcript by concatenating words list
    transcript = " ".join(words_list).lstrip(punctuation).strip()

    # Runs text processing functions on each and every transcript
    logger.info(f"Text Processing has been started")
    processed_transcript = TextProcessing.process_transcript(transcript=transcript)
    logger.info(f"Text Processing has been completed")

    return count + start_index, processed_transcript


def get_temp_store_path(file_name: str):
    """This function returns temporary store path
    for storing ffmpeg mp4 output files
    """

    temp_store_path = os.sep.join([constants.TMP_DIR, os.path.basename(file_name)])
    return temp_store_path


def clear_directory(path: str):
    """This function clears temporary storage
    after every iteration
    """
    if os.path.exists(path):
        logger.info(f"Started Removing Temp file {path}")
        os.remove(path)
        logger.info(f"Successfully removed temp file {path} from {constants.TMP_DIR}")
    else:
        logger.info(f"Temp file --> {path} doesn't exist, skipping removal")


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
