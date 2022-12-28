import json
import logging
import signal
from datetime import datetime

import constants
from exceptions import DownstreamAPIError, NoneError
from utils import handle_failure, timeout_handler

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
# Sends a signal to timeout_handler which in turn raises Timeout Exception
signal.signal(signal.SIGALRM, timeout_handler)


def sf_failure_handler(event, context):
    """Handles step function timeouts, long running tasks and failures"""
    try:
        # Sends remaining time as an alarm to signal
        signal.alarm(int(context.get_remaining_time_in_millis() / 1000) - 1)
        # Get's state machine details value by parsing event dict
        sf_details = event.get(constants.STEP_FUNCTION_DETAIL)
        # Let's get step function metadata to handle failure/timeouts
        status = sf_details.get(constants.STATUS)
        execution_arn = sf_details.get(constants.EXECUTION_ARN)
        step_function_input = json.loads(sf_details.get(constants.INPUT))
        masked_request_id = step_function_input.get(constants.MASKED_REQUEST_ID)

        # Let's log some info for further monitoring
        logger.info(f"State machine execution ARN is --> {execution_arn}")
        logger.info(
            f"Status of state machine corresponding to {masked_request_id} --> {status}"
        )
        logger.info(
            f"State machine input corresponding to {masked_request_id} --> {step_function_input}"
        )

        if step_function_input and status in constants.FAILURE_STATUS:
            logger.info(f"Handling failed state-machine workflow --> {execution_arn}")
            handle_failure(masked_request_id=masked_request_id)
            logger.info(
                f"Handled failed state-machine workflow --> {execution_arn} successfully"
            )

        elif not step_function_input and status in constants.FAILURE_STATUS:
            logger.info(
                f"Failed to read input from state machine execution {execution_arn}, skipping status updation in DB"
            )
            logger.error(
                "Step function input object not constructed, Cannot access a 'None' object"
            )
            # Raises Value/Type Error if status_response is not present, will be handled at workflow
            raise NoneError(
                "Step function input object not constructed, Cannot access a 'None' object"
            )
        # Sends a success alram to signal
        signal.alarm(0)
    except (RuntimeError, MemoryError, AttributeError, TypeError, ValueError):
        logger.error(
            f"Exception occured while trying to handle state-machine failed workflow"
        )
        # Sends a success alram to signal
        signal.alarm(0)
        # Raises Runtime/Memory/Value Errors which will be handled at workflow
        raise DownstreamAPIError(
            f"Exception occured while trying to handle state-machine failed workflow"
        )
