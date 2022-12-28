import logging

from boiler_plate.utility import constants
from boiler_plate.utility.utils import (create_error_response,
                                        write_to_output_location)

logger = logging.getLogger("file")

def scheduler_config():
    """Initializes apscheduler object using params from constants module"""

    # Get's executor and job_store from constants module
    executor = constants.EXECUTORS.get(constants.EXECUTOR_TYPE)
    job_store = constants.JOB_STORES.get(constants.JOB_STORE_TYPE)

    # Creates scheduler_config object using apscheduler params in constants module
    scheduler_config = {
        constants.JOB_STORE: {
            constants.JOB_STORE_KEY: constants.JOB_STORE_TYPE,
            constants.JOB_STORE_URL: job_store.get(constants.JOB_STORE_URL),
        },
        constants.EXECUTOR: {
            constants.EXECUTOR_KEY: executor.get(constants.EXECUTOR_VALUE),
            constants.MAX_WORKERS: executor.get(constants.MAX_WORKERS),
        },
        constants.COALESCE_KEY: constants.COALESCE_VALUE,
    }

    # Returns scheduler config object
    return scheduler_config
    
def handle_failure(
    file_obj: object,
    request_id: str,
    model_name: str,
    response_path: str = None,
    write_output: bool = True,
):
    """Function to handle failure and to generate error response"""

    logger.info(
        f"Pratilipi Speech To Text/Diarization Execution for {request_id} is failed/In Progress"
    )
    # Creates error response dictionary using constants module
    logger.info(f"Creating {model_name.upper()} error response")
    error_response = create_error_response(
        status=constants.ERROR_KEY, model=constants.MODEL_NAMES[model_name]
    )
    logger.info(
        f"{model_name.upper()} error response created successfully --> {error_response}"
    )
    if write_output:
        write_to_output_location(output_path=response_path, output=error_response)

    # Updates DB status to Error
    if file_obj.status != constants.ERROR_DB:
        logger.info(f"Updating Database status of {request_id}")
        file_obj.status = constants.ERROR_DB
        file_obj.save()
        logger.info(
            f"Database status of {request_id} updated successfully to {constants.ERROR_DB}"
        )
