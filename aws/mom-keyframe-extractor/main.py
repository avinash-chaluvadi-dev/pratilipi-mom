import json
import logging
import os
import signal

from dotenv import dotenv_values
from exceptions import DownstreamAPIError, NoneError
from utils import sync_get, timeout_handler

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Creates an OrderedDict of env variables from .env file
CONFIG = dict(dotenv_values(".env"))
# Sends a signal to timeout_handler which in turn raises Timeout Exception
signal.signal(signal.SIGALRM, timeout_handler)


def keyframe_ext_handler(event, context):
    """Handler function for invoking Framify endpoint running on WorkOS"""

    # Fetches masked request id from event
    masked_request_id = event.get("masked_request_id")
    logger.info(
        f"Triggering keyframe extractor job corresponding to {masked_request_id}"
    )
    try:
        # Sends remaining time as an alarm to signal
        signal.alarm(int(context.get_remaining_time_in_millis() / 1000) - 1)
        # Checks whether masked request id is being passed to event from state or not
        if masked_request_id:
            # Fetches Keyframe extractor URL from environment variable
            keyframe_ext_url = CONFIG.get("KEYFRAME_EXT_URL")
            # Invokes sync_get function which in turn invokes Framify API
            keyframe_ext_response = sync_get(
                url=f"{keyframe_ext_url}/{masked_request_id}/",
                ssl_verify=CONFIG.get("SSL_VERIFY"),
            )
            logger.info(
                f"Keyframe extractor job triggered successfully --> {keyframe_ext_response}"
            )

            # Sends a success alram to signal
            signal.alarm(0)
            # Returns keyframe extractor response to step function
            return {
                "headers": {"Content-Type": "application/json"},
                "statusCode": 200,
                "isBase64Encoded": False,
                "body": keyframe_ext_response,
            }
        elif not masked_request_id:
            logger.error(
                f"Request ID object not constructed, Cannot access a 'None' object"
            )
            # Sends a success alram to signal
            signal.alarm(0)
            # Raises Value/Type Error if masked_request_id is not present, will be handled at workflow
            raise NoneError(
                "Request ID object not constructed, Cannot access a 'None' object"
            )

    except (RuntimeError, MemoryError, AttributeError, TypeError, ValueError):
        logger.error(
            f"Exception occured while trying to invoke Keyframe Extractor engine"
        )
        # Sends a success alram to signal
        signal.alarm(0)
        # Raises Runtime/Memory/Value Errors which will be handled at workflow
        raise DownstreamAPIError(
            f"Exception occured while trying to invoke Keyframe Extractor engine"
        )
