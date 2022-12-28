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


def label_cls_handler(event, context):
    """Handler function for invoking Label CLS endpoint running on WorkOS"""

    # Fetches masked request id from event
    masked_request_id = event.get("masked_request_id")
    logger.info(f"Triggering label classifier job corresponding to {masked_request_id}")
    try:
        # Sends remaining time as an alarm to signal
        signal.alarm(int(context.get_remaining_time_in_millis() / 1000) - 1)
        # Checks whether masked request id is being passed to event from state or not
        if masked_request_id:
            # Fetches Label CLS URL from environment variable
            label_cls_url = CONFIG.get("LABEL_CLS_URL")
            # Invokes sync_get function which in turn invokes label cls API
            label_cls_response = sync_get(
                url=f"{label_cls_url}/{masked_request_id}/",
                ssl_verify=CONFIG.get("SSL_VERIFY"),
            )
            logger.info(
                f"Label classifier job triggered successfully --> {label_cls_response}"
            )

            # Sends a success alram to signal
            signal.alarm(0)
            # Returns label classifier response to step function
            return {
                "headers": {"Content-Type": "application/json"},
                "statusCode": 200,
                "isBase64Encoded": False,
                "body": label_cls_response,
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
            f"Exception occured while trying to invoke Label Classifier engine"
        )
        # Sends a success alram to signal
        signal.alarm(0)
        # Raises Runtime/Memory/Value Errors which will be handled at workflow
        raise DownstreamAPIError(
            f"Exception occured while trying to invoke Label Classifier engine"
        )
