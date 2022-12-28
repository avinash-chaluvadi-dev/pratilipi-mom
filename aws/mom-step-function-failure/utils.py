import logging
from distutils.util import strtobool

import constants
import requests
import urllib3
from dotenv import dotenv_values
from exceptions import LambdaTimeoutError, RequestError

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# To disable/supress unverified https warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Creates an OrderedDict of env variables from .env file
CONFIG = dict(dotenv_values(".env"))


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
        raise RequestError(f"Timeout error while requesting {url}")

    except requests.exceptions.TooManyRedirects:
        logger.error(f"Too many redirects while requesting {url}")
        raise RequestError(f"Too many redirects while requesting {url}")

    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error while connecting to {url}")
        raise RequestError(f"Connection error while connecting to {url}")

    except Exception as e:
        logger.error(
            "Unable to get url {} due to {}.".format(
                url, getattr(e, "message", repr(e))
            )
        )
        raise LambdaTimeoutError(
            "Unable to get url {} due to {}.".format(
                url, getattr(e, "message", repr(e))
            )
        )


def file_status_update(status: str, masked_request_id: str):
    """Updates the status of api_file table based on request_id/file_name"""

    summarizer_url = CONFIG.get("SUMMARIZER_URL")
    logger.info(f"MOM Summarizer base endpoint --> {summarizer_url}")
    # Creates upload endpoint from summarizer base url and constants module
    file_status_url = f"{summarizer_url}/{constants.STATUS_UPDATE_SUFFIX}"
    logger.info(
        f"Endpoint for updating file_status to ({status}) --> {file_status_url}"
    )
    logger.info(
        f"Updating the status of {masked_request_id} to {status} in '(api_file)' table"
    )

    # Creates data dict for file-status update API(POST)
    data = {
        constants.STATUS: status,
        constants.MASKED_REQUEST_ID: masked_request_id,
    }

    # Invokes sync_postfunction which in turn invokes file-status update API
    status_response = sync_post(url=file_status_url, data=data)

    # Log statements for further monitoring
    logger.info(f"File Status update response --> {status_response}")
    logger.info(f"Successfully updated the status of {masked_request_id} to {status}")


def handle_failure(masked_request_id: str):
    """Handles timeout/error stages of state machine
    by updating DB status to Error
    """

    # Invokes file_status_update function to update the DB status
    file_status_update(
        status=constants.ERROR_STATUS, masked_request_id=masked_request_id
    )


def timeout_handler(_signal, _frame):
    """Lambda timout handler function"""
    raise LambdaTimeoutError("Lambda time limit exceeded")
