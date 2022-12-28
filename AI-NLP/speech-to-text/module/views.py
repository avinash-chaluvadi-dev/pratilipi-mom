"""
    @Author         : HIVE TEAM
    @Purpose        : SentimentClassifier Standard Views***
    @Description    :
    @Date           : 20-08-2021
    @Last Modified  : 20-08-2021
"""
import json
from pathlib import Path

import pickle
import numpy as np

from django.shortcuts import render, get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.response import Response

from module.speech2text import config
from module.speech2text import model_serve
from module.speech2text.utils import utils_tools

from output_endpoint import output_endpoint
from module import specs
from rest_api.models import File

from boiler_plate.utility.utils import get_file_base_path
from boiler_plate.utility.utils import write_to_output_location


def api_func(inp):
    with open(inp, "r") as f:
        json_data = json.load(f)
    resp = model_serve(json_data)
    return resp


@api_view(["GET"])
def speech_to_text(request, *args, **kwargs):
    request_id = kwargs.get("request_id")

    file_obj = get_object_or_404(File, masked_request_id=request_id)

    base_path = get_file_base_path(
        file_obj.get_file().path,
        specs.get_input_location(),
        file_obj.get_file_name(),
    )

    path = Path(f"{base_path}/speaker_diarization.json")

    resp = api_func(path)

    Path(f"{base_path}/{specs.get_output_location()}").mkdir(parents=True, exist_ok=True)
    op = str(file_obj.get_file().path).replace(file_obj.get_file_name(), "")
    if resp["status"] == 'success':
        write_to_output_location(op, specs.get_output_location(), "speech_to_text.json", resp)
        return Response(resp)
    else:
        return Response({"status": "fail"})
