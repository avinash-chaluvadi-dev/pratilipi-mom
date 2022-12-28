import json
import logging
from datetime import datetime, timedelta
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


def sync_get(url: str, params: dict):
    """Synchronous Http GET method for Status API"""

    # Fetches ssl verify value from environment variable
    ssl_verify = bool(strtobool(CONFIG.get("SSL_VERIFY")))

    # Invokes Status Endpoint using python requests library
    try:
        return requests.get(url=url, data=json.dumps(params), verify=ssl_verify).json()

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


def get_status_url(lambda_event: dict):
    """Generates Status Endpoint dynamically based on event dictionary"""

    # Fetches module from lambda event
    module = lambda_event.get(constants.MODULE)
    # Fetches component from lambda event
    component = lambda_event.get(constants.COMPONENT)
    # Fetches response path from lambda event
    response_path = lambda_event.get(constants.RESPONSE_PATH)
    # Creates params dict for status API
    params = {constants.RESPONSE_PATH: response_path}

    # Checks whether masked request id is being passed to event from state or not
    if module and component:
        # Fetches MOM base URL from environment variable
        mom_base_url = CONFIG.get("MOM_BASE_URL")
        # Creates status URL by combining base url and mom components
        logger.info("Generating status endpoint using params from event dictionary")
        status_url = f"{mom_base_url}/{module}/{constants.REST_API}/{component}/{constants.STATUS}/"
        logger.info(f"Status endpoint generated successfully --> {status_url}")

        # Returns Status URL/Endpoint
        return status_url, params


def is_job_crashed(backend_start_time: str):
    """Calculates timedelta of backend_start_time between current
    datetime and returns whether JOB has crashed or not
    """

    # Converts string into datetime.datetime object
    backend_start_time_obj = datetime.strptime(
        backend_start_time, constants.BACKEND_TIME_FORMAT
    )

    # Calculates time difference between backend_start_time and present datetime
    time_difference = (
        backend_start_time_obj
        + timedelta(minutes=int(CONFIG.get("TIMEOUT")))
        - datetime.now()
    )

    # Returns status as CRASHED if time difference is greather than TIMEOUT
    if time_difference.days == -1 or time_difference.seconds >= 0:
        return True

    else:
        return False


def timeout_handler(_signal, _frame):
    """Lambda timout handler function"""
    raise LambdaTimeoutError("Lambda time limit exceeded")
