import logging
import signal

from utils import S3Adapter, get_bucket_name, timeout_handler

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Sends a signal to timeout_handler which in turn raises Timeout Exception
signal.signal(signal.SIGALRM, timeout_handler)


def playback_handler(event, context):
    """
    Entrypoint function for generating a presigned/public URI
    which inturn helps to access content of private S3 Bucket
    """
    try:
        # Sends remaining time as an alarm to signal
        signal.alarm(int(context.get_remaining_time_in_millis() / 1000) - 1)
        # Parsing query parameters from event
        query_params = event.get("queryStringParameters")
        # Getting file_uri/input to aws transcribe
        file_uri = query_params.get("file_uri")
        if not file_uri:
            # Creates playback error message string which will be used in error response
            error_message = "No file specified for generating presigned URI, file_uri object not constructed"
            logger.error(error_message)
            # Sends a success alram to signal
            signal.alarm(0)
            # Returns playback error response in form of Json
            return {"statusCode": 400, "isBase64Encoded": False, "body": error_message}

        else:
            logger.info(f"Input media file URI is --> {file_uri}")
            # Get bucket name from either env variables or from S3 URL
            bucket_name = get_bucket_name(s3_uri=file_uri)
            logger.info(
                f"Bucket for accessing media file {file_uri} is --> {bucket_name}"
            )
            # Initializes s3_utils object using which we can access s3 utility functions
            logger.info(
                "Initializing S3Adapter object for accessing s3 utility functions"
            )
            s3_utils = S3Adapter(bucket_name=bucket_name)
            s3_source_signed_uri = s3_utils.generate_presigned_uri(path=file_uri)
            # Sends a success alram to signal
            signal.alarm(0)
            # Returns the generated s3_source_signed_uri of input file_uri
            return {
                "statusCode": 200,
                "isBase64Encoded": False,
                "body": s3_source_signed_uri,
            }

    except (RuntimeError, MemoryError, AttributeError, TypeError, ValueError):
        logger.error("Exception occured while trying to invoke playback engine")
        # Sends a success alram to signal
        signal.alarm(0)
        # Returns playback error response in form of Json
        return {
            "statusCode": 400,
            "isBase64Encoded": False,
            "body": "Exception occured while trying to invoke playback engine",
        }
