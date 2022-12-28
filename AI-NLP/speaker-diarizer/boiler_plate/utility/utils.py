# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Global Constant File***
    @Description    :
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

import json
import logging as lg
import os
import pickle
from pathlib import Path
import math
from django.conf import settings

import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from module import specs

logger = lg.getLogger("file")


def handle_uploaded_files(filename: str, path: str) -> None:
    """function for handle intake files"""
    with open(path, "wb+") as destination:
        for chunk in filename.chunks():
            destination.write(chunk)


def get_input_file_as_dict(file: object) -> dict:
    """Returns file content as a dict"""
    with file.open(mode="rb") as stream:
        input_file = json.load(stream)

    return input_file


def write_to_output_location(
        base_path: str, output_location: str, file_name: str, output: dict
) -> None:
    """Writes the output response to corresponing output location

    Args:
        :base_path: base path of the meeting ID
        :output_location: folder name in which content should be written
        :file_name: name of the file
        :output: output response that needs to written

    """

    logger.info(f"started process to write outptu file for  {file_name}")
    # create output dir if does not exits
    Path(f"{base_path}/{output_location}/").mkdir(parents=True, exist_ok=True)

    # Generate output path
    output_path = Path(f"{base_path}/{output_location}/{file_name}")
    logger.info(f"output path is {output_path}")

    # Writing output to corresponding file.
    with open(output_path, "w") as destination:
        destination.write(json.dumps(output, indent=4))


def get_file_base_path(file_path: str, specs_path: str, file_name: str) -> str:
    """Returns base path of a file"""
    base_path = str(file_path).replace(file_name, "")
    # print("specs_path : ", specs_path)
    # print("base_path : ", base_path)
    if specs_path:
        base_path = f"{base_path}/{specs_path}"

    logger.info(f"base path for file {file_name} is {base_path}")

    return base_path


def covert_mp4_to_slice(
        base_path: str, path: str, resp: dict, extn: str
) -> list:
    segment_dict = pickle.loads(
        resp["Speaker_Diarizer_Result"]["Details"]["dataframe"]
    )

    Path(f"{base_path}/{specs.get_output_location()}/").mkdir(
        parents=True, exist_ok=True
    )

    for i in range(len(segment_dict["start_time"])):
        ffmpeg_extract_subclip(
            f"{path}",
            math.floor(segment_dict["start_time"][i] / 1000),
            math.floor(segment_dict["stop_time"][i] / 1000),
            targetname=f"{base_path}/{specs.get_output_location()}/chunk_{math.floor(segment_dict['start_time'][i])}_{math.floor(segment_dict['stop_time'][i])}_{int(segment_dict['label'][i])}.{extn}",
        )

    return True


def response_decoder(
        base_path: str, resp: dict, video_flag: bool, extn: str
) -> dict:
    """response decoder, stores the dataframe and the audio chunks to the system.
    Extract the audio files corresponding to users.
    Decode and save csv.
    NOTE : The csv_path and the chunk_storage paths can be configured by the api team, depending upon its usage.
    """

    df = pickle.loads(resp["Speaker_Diarizer_Result"]["Details"]["dataframe"])
    df.columns = ["chunk_file", "start_time", "end_time", "speaker_id"]
    df["chunk_id"] = np.arange(df.shape[0]) + 1

    media_root = settings.MEDIA_ROOT

    def full_path(x):
        op_path = str(
            Path(f"{base_path}/{specs.get_output_location()}/").joinpath(x)
        )
        # print("1211111111111111111 :", Path(media_root).as_uri())
        #return op_path.replace(os.path.abspath(media_root), "http://localhost:8000/media/")
        return op_path

    def generate_video_path(x):
        x = x.split(".")[0] + "." + extn
        op_path = str(
            Path(f"{base_path}/{specs.get_output_location()}/").joinpath(x)
        )
        #return op_path.replace(os.path.abspath(media_root), "http://localhost:8000/media/")
        return op_path

    df["audio_path"] = df["chunk_file"].apply(full_path)

    if video_flag:
        df["video_path"] = df["chunk_file"].apply(generate_video_path)

    segment_dict = resp["Speaker_Diarizer_Result"]["Details"]["audio_chunks"]
    Path(f"{base_path}/{specs.get_output_location()}").mkdir(
        parents=True, exist_ok=True
    )
    df.to_csv(Path(f"{base_path}/{specs.get_output_location()}/sample.csv"))
    for spk in segment_dict.keys():
        for chunk_name in segment_dict[spk].keys():
            audio_file = pickle.loads(segment_dict[spk][chunk_name])
            audio_file.export(
                Path(
                    f"{base_path}/{specs.get_output_location()}/{chunk_name}"
                ),
                format="wav",
            )
    response = {
        "status": "success",
        "model": "speaker_diarization",
        "response": df.to_dict("records"),
    }
    write_to_output_location(
        base_path,
        specs.get_output_location(),
        "speaker_diarization.json",
        response,
    )
    return df.to_dict("records")
