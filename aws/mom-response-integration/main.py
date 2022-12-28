import logging

import constants
from dotenv import dotenv_values
from utils import file_status_update, sync_get

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# Creates an OrderedDict of env variables from .env file
CONFIG = dict(dotenv_values(".env"))


def integration_handler(event, context):
    """Handler function for invoking Response Integration endpoint running on WorkOS"""

    # Fetches masked request id from lambda event
    masked_request_id = event.get("masked_request_id")

    # Fetches keyframes parameter from lambda event
    keyframes = event.get(constants.KEYFRAMES)

    # Checks whether masked request id is being passed to event from state or not
    if masked_request_id:
        # Fetches Response Integration URL from environment variable
        integration_url = CONFIG.get("INTEGRATION_URL")
        # Creates query params, which will be passed to response_integration API
        params = {constants.KEYFRAMES: keyframes}
        # Invokes sync_get function which in turn invokes response integration API
        integration_response = sync_get(
            url=f"{integration_url}/{masked_request_id}/", params=params
        )

        # Let's update the status of api_file to Ready For Review if integration is successful, otherwise returns Error
        if not constants.ERROR in integration_response:
            # Calls file_status_update function to update status of api_file table
            status = file_status_update(masked_request_id=masked_request_id)
            return {
                "headers": {"Content-Type": "application/json"},
                "statusCode": 200,
                "status": status,
                "isBase64Encoded": False,
                "body": integration_response,
            }
        else:
            return {
                "headers": {"Content-Type": "application/json"},
                "statusCode": 200,
                "status": constants.ERROR.capitalize(),
                "isBase64Encoded": False,
                "body": integration_response,
            }
