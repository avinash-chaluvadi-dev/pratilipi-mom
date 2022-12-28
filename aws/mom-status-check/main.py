import json
import logging
import os
import signal

import constants
from exceptions import DownstreamAPIError, NoneError
from utils import get_status_url, is_job_crashed, sync_get, timeout_handler

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
# Sends a signal to timeout_handler which in turn raises Timeout Exception
signal.signal(signal.SIGALRM, timeout_handler)


def status_handler(event, context):
    """Handler function for invoking Status endpoint running on WorkOS"""
    try:
        # Sends remaining time as an alarm to signal
        signal.alarm(int(context.get_remaining_time_in_millis() / 1000) - 1)
        # Get's status API endpoint using get_status_url
        status_url, status_params = get_status_url(lambda_event=event)
        # Invokes sync_get function which in turn invokes Status API
        status_response = sync_get(url=status_url, params=status_params)
        if not status_response:
            logger.error(
                f"Status response object not constructed, Cannot access a 'None' object"
            )
            # Sends a success alram to signal
            signal.alarm(0)
            # Raises Value/Type Error if status_response is not present, will be handled at workflow
            raise NoneError(
                "Status response object not constructed, Cannot access a 'None' object"
            )

        else:
            # Parses status API response to get status key
            status = status_response.get("status")
            logger.info(
                f"Checking whether {event.get(constants.COMPONENT)} has been {constants.CRASHED_STATUS}/{constants.IN_PROGRESS_STATUS}/{constants.COMPLETED_STATUS}"
            )
            logger.info(
                f"Status of {event.get(constants.COMPONENT).capitalize()} job corresponding to {event.get(constants.REQUEST_ID_KEY)} is {status}"
            )
            if status == constants.IN_PROGRESS_STATUS:
                # Invokes is_job_crashed func to check whether job is running/crashed
                if is_job_crashed(
                    backend_start_time=event.get(constants.BACKEND_START_TIME)
                ):
                    logger.info(
                        f"{event.get(constants.COMPONENT).capitalize()} job corresponding to {event.get(constants.REQUEST_ID_KEY)} has been crashed due to memory issue, new job will be triggered automatically"
                    )
                    status = constants.CRASHED_STATUS
                else:
                    logger.info(
                        f"{event.get(constants.COMPONENT).capitalize()} job corresponding to {event.get(constants.REQUEST_ID_KEY)} is still {constants.IN_PROGRESS_STATUS}"
                    )
                    status = constants.IN_PROGRESS_STATUS
            elif status == constants.COMPLETED_STATUS:
                logger.info(
                    f"{event.get(constants.COMPONENT).capitalize()} job corresponding to {event.get(constants.REQUEST_ID_KEY)} has been {status}"
                )
            # Sends a success alram to signal
            signal.alarm(0)
            # Returns response dict with corresponding status
            return {
                "headers": {"Content-Type": "application/json"},
                "statusCode": 200,
                "isBase64Encoded": False,
                "body": status,
            }
    except (RuntimeError, MemoryError, AttributeError, TypeError, ValueError):
        logger.error(f"Exception occured while trying to invoke status engine")
        # Sends a success alram to signal
        signal.alarm(0)
        # Raises Runtime/Memory/Value Errors which will be handled at workflow
        raise DownstreamAPIError(
            f"Exception occured while trying to invoke status check engine"
        )
