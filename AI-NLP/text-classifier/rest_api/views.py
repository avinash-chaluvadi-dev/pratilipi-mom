import json
import logging as lg

from rest_framework.decorators import action, api_view
from rest_framework.response import Response

from boiler_plate.utility import utils

logger = lg.getLogger("file")


@api_view(["GET"])
def status_check(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper for checking the status of ML models
    """

    # Fetching model_name from kwargs
    model_name = kwargs.get("model")
    logger.info(f"Checking status of {model_name.capitalize()} job")

    response_dict = json.loads(request.body)
    response_path = response_dict[list(response_dict.keys())[0]]
    logger.info(f"Loading {model_name.capitalize()} output json from {response_path}")
    logger.info(f"{model_name.capitalize()} output json loaded successfully")

    status_response = utils.model_status(file_path=response_path, model_name=model_name)
    logger.info(f"{model_name.capitalize()} status response --> {status_response}")

    return Response(status_response)
