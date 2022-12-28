"""
    @Author         : HIVE TEAM
    @Purpose        : SentimentClassifier Standard Views***
    @Description    :
    @Date           : 20-08-2021
    @Last Modified  : 20-08-2021
"""

from pathlib import Path

from django.shortcuts import get_object_or_404
from pydub import AudioSegment
from rest_framework.decorators import api_view
from rest_framework.response import Response

from boiler_plate.utility.utils import (
    covert_mp4_to_slice,
    get_file_base_path,
    response_decoder,
)
from module import specs
from rest_api.models import File

from .speaker_diarization import spk_diarize_serve

# {base_path}/{specs.get_output_location()}


@api_view(["GET"])
def speaker_diarization(
    request: object, *args: list, **kwargs: dict
) -> Response:
    """
    API wrapper over Topic extractor Model
    """

    video_exts = [
        "mp4",
        "mkv",
    ]
    # Fetching input file location using the REQUEST_ID
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    base_path = get_file_base_path(
        file_obj.get_file().path,
        specs.get_input_location(),
        file_obj.get_file_name(),
    )
    output_path = f"{base_path}/uploaded_file.wav"
    path = Path(f"{base_path}/{file_obj.get_file_name()}")
    file_name = file_obj.get_file_name()

    flag = False
    audio = AudioSegment.from_file(path, file_name.split(".")[-1])
    audio = audio.set_frame_rate(16000)
    inputfile = audio.export(output_path, format="wav")

    if file_name.split(".")[-1] in video_exts:
        flag = True

    resp = spk_diarize_serve(inputfile)
    video_flag = False
    if flag:
        video_flag = covert_mp4_to_slice(
            base_path, path, resp, file_name.split(".")[-1]
        )

    if resp["Speaker_Diarizer_Result"]["status"] == "Success":
        output = response_decoder(
            base_path, resp, video_flag, file_name.split(".")[-1]
        )
    else:
        return Response({"status": "fail"})
    return Response(
        {
            "status": "success",
            "model": "speaker_diarization",
            "response": output,
        }
    )
