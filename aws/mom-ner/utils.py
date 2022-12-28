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


def sync_get(url: str, ssl_verify: str):
    """Synchronous Http GET method for Entity recognizer API"""

    # Initializes an empty dictionary for response
    response_dict = {}

    # Fetches ssl verify value from environment variable
    ssl_verify = bool(strtobool(CONFIG.get("SSL_VERIFY")))

    # Invokes NER Endpoint using python requests library
    try:
        response = requests.get(url=url, verify=ssl_verify).json()
        response_dict[constants.MODULE] = constants.NER_MODULE
        response_dict[constants.COMPONENT] = constants.NER_COMPONENT
        response_dict[constants.RESPONSE_PATH] = response.get(list(response.keys())[0])
        return response_dict

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


def timeout_handler(_signal, _frame):
    """Lambda timout handler function"""
    raise LambdaTimeoutError("Lambda time limit exceeded")
