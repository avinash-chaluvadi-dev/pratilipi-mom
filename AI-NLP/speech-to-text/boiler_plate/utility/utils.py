# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Global Constant File***
    @Description    :
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""
import json
from pathlib import Path
import logging as lg

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

    if specs_path:
        base_path = f"{base_path}/{specs_path}"

    logger.info(f"base path for file {file_name} is {base_path}")

    return base_path
