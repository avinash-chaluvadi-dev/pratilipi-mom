import logging
import re

import boto3
from botocore import exceptions
from botocore.client import Config
from dotenv import dotenv_values

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Creates an OrderedDict of env variables from .env file
CONFIG = dict(dotenv_values(".env"))


def get_bucket_name(s3_uri: str = None) -> str:
    """Returns bucket_name from either constants module or S3 URI"""

    # Checks whether constants module has BUCKET_NAME variable or not
    if CONFIG.get("BUCKET_NAME"):
        bucket_name = CONFIG.get("BUCKET_NAME")

    else:
        if s3_uri:
            # Extracts bucket name from S3 URI
            bucket_name, _ = s3_uri.replace("s3://", "").split("/", 1)

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


class S3Adapter:
    def __init__(self, bucket_name: str):
        """Constructor to initialize the state of object from method params"""

        # Instantiates the state of object from the input params
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource(
            "s3", config=Config(signature_version=CONFIG.get("SIGNATURE_VERSION"))
        )

    def get_prefix_from_uri(self, path: str) -> str:
        """
        To generate the prefix from URI
        """
        # Generates the absolute path from root bucket and cleans using clean_path function
        return clean_path(
            path.replace(CONFIG.get("S3_URI"), "").replace(self.bucket_name, "").strip()
        )

    def generate_presigned_uri(self, path):
        """
        Generates presigned/temporary URI of
        corresponding file object present in S3 Bucket
        """
        try:
            # Let's add some log statements for further monitoring
            logger.info(f"Generating presigned URI of audio file --> {path}")

            if CONFIG.get("S3_URI") in path:
                # Removes the S3 URI headers from read_path
                path = self.get_prefix_from_uri(path=path)
            # Generates source signed uri/public uri which expires after the specified expiry time
            s3_source_signed_uri = self.s3_resource.meta.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": path},
                ExpiresIn=CONFIG.get("SIGNED_URL_TIMEOUT"),
            )
            logger.info(
                f"Presigned URI generated successfully --> {s3_source_signed_uri}"
            )
            return s3_source_signed_uri

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)


def timeout_handler(_signal, _frame):
    """Lambda timout handler function"""
    logger.error("Lambda time limit exceeded")
    # Sends a success alram to signal
    signal.alarm(0)
    # Returns playback error response in form of Json
    return {
        "statusCode": 400,
        "isBase64Encoded": False,
        "body": "Lambda time limit exceeded",
    }
