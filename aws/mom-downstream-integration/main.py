import json
import logging
from datetime import datetime

import constants
from utils import (backend_start_time_update, file_status_update,
                   get_masked_request_id, get_state_machine_input,
                   run_keyframes, start_step_function_execution)

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def downstream_handler(event, context):
    """Invokes step function workflow which in turn invokes downstrem API's"""

    # Get message which is sent to MOM Downstream SNS Topic by diarization
    message = json.loads(event["Records"][0]["Sns"]["Message"])
    logger.info("Message received successfully from mom-downstream-integration topic")
    logger.info(f"Downstream Integration Message --> {message}")

    # Extracts media_path and s2t_output_path from SNS message
    media_path = message.get("media_path")
    s2t_output_path = message.get("s2t_output_path")
    diarization_output_path = message.get("diarization_output_path")

    # Let's logs input file path to aws transcribe and speech to text output path
    logger.info(f"Diarization Input file path --> {media_path}")
    logger.info(f"Speech to text output path --> {s2t_output_path}")
    logger.info(f"Diarization output path --> {diarization_output_path}")

    # Extracts diarization and media_path extesion from SNS message
    extension = message.get("extension")
    diarization = int(message.get("diarization"))
    logger.info(f"Input Media Type --> {extension}")
    logger.info(f"Diarization is {bool(int(diarization))}")

    # run_keyframes function decides whether or not keyframe MS to be performed
    keyframes = run_keyframes(extension=extension, diarization=diarization)

    # Get speech to text success, if status is error function exits
    status = message.get(constants.STATUS)

    # Try and except block to catch step function invoke error
    try:
        if status == constants.SUCCESS:
            # Get masked request id by invoking upload api in mom-summarizer microservice
            masked_request_id_res = get_masked_request_id(media_uri=media_path)
            if len(masked_request_id_res) != 0:
                masked_request_id = masked_request_id_res[0].get(
                    constants.REQUEST_ID_KEY
                )
                logger.info(
                    f"Masked Request ID corresponding to {media_path} --> {masked_request_id}"
                )

                # Get's present datetime object to update backend_start_time in api_file
                backend_start_time = datetime.now()

                # Creates state machine input dictionary using get_state_machine_input func
                state_input = get_state_machine_input(
                    keyframes=keyframes,
                    wait_time=constants.WAIT_TIME,
                    masked_request_id=masked_request_id,
                    backend_start_time=backend_start_time,
                )
                logger.info(f"State Machine input --> {state_input}")

                # Starts the execution of state-machine using boto3
                logger.info("Started invoking {mom-state-machine-sit} State Machine")
                response = start_step_function_execution(state_input=state_input)

                # Log statements for further monitoring
                logger.info("State Machine invoked successfully")
                logger.info(
                    f"State Machine success invocation confirmation --> {response}"
                )

                # Calls file_status_update function to update backend_start_time of api_file table
                update_response = backend_start_time_update(
                    backend_start_time=backend_start_time, file_name=media_path
                )

            else:
                logger.error(
                    f"{media_path} is not present in {constants.FILE_TABLE} in {constants.DB_NAME}, updating the status to {constants.ERROR_STATUS}"
                )
                # Calls file_status_update function to update status of api_file table
                update_response = file_status_update(
                    status=constants.ERROR_STATUS, file_name=media_path
                )

        else:
            logger.error(
                "Speaker Diarization/Speech To Text Failed, skipping execution of State Machine"
            )

            # Calls file_status_update function to update status of api_file table
            update_response = file_status_update(
                status=constants.ERROR_STATUS, file_name=media_path
            )

    # Excepts any RuntimeError/MemoryError/TypeError and updates the DB status to error
    except (RuntimeError, MemoryError, ValueError, TypeError):
        logger.error("Exception occured while running Downstream Engine")
        # Calls file_status_update function to update status of api_file table
        update_response = file_status_update(
            status=constants.ERROR_STATUS, file_name=media_path
        )
