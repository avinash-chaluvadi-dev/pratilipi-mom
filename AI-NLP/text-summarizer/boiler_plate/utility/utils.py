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
import re
from pathlib import Path
from typing import Callable

import aiohttp
import boto3
from asgiref.sync import sync_to_async
from botocore import exceptions
from django.conf import settings

from boiler_plate.utility import constants
from rest_api.models import File, MLModelStatus

logger = lg.getLogger("file")


class S3Utils:
    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")
        self.bucket_name = settings.AWS_STORAGE_BUCKET_NAME

    def load_json(self, load_prefix):
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=load_prefix)
            return json.loads(obj["Body"].read().decode("utf-8"))

        except exceptions.DataNotFoundError as error:
            logger.error("Specified prefix does not exist")
            raise RuntimeError(error)

    def write_json(self, store_prefix: str, content: dict) -> None:
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=store_prefix, Body=content
            )
            logger.info(f"Response saved successfully into {store_prefix}")

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def delete_object(self, file_prefix: str):
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_prefix)
            logger.info(f"{file_prefix} deleted successfully from {self.bucket_name}")

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def prefix_exist(self, file_prefix):
        """Function to check whether prefix exist of not"""
        try:
            # Creates bucket object from s3 resource
            bucket = self.s3_resource.Bucket(self.bucket_name)
            logger.info(f"Started checking whether {file_prefix} exist or not")
            if any(
                [
                    obj.key == file_prefix
                    for obj in list(bucket.objects.filter(Prefix=file_prefix))
                ]
            ):
                return True
            else:
                return False

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def copy(self, copy_source: str, key: str):
        """Function to copy object from source to destination"""
        try:
            self.s3_client.copy_object(
                Bucket=self.bucket_name, CopySource=copy_source, Key=key
            )

        except exceptions.ClientError as error:
            logger.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def get_absolute_path(self, file_prefix: str) -> str:
        return f"{constants.S3_URI}{self.bucket_name}/{file_prefix}"

    def parse_s3_uri(self, s3_uri: str):
        return s3_uri.replace(constants.S3_URI, "").split("/", 1)[1]


def model_status(file_path: str, model_name: str):
    if settings.USE_S3:
        s3_utils = S3Utils()
        response_prefix = s3_utils.parse_s3_uri(s3_uri=file_path)
        output_exist = s3_utils.prefix_exist(file_prefix=response_prefix)

        if output_exist:
            output_response = s3_utils.load_json(load_prefix=response_prefix)
            if (
                output_response.get(constants.STATUS_KEY)
                == constants.IN_PROGRESS_STATUS
            ):
                logger.info(
                    f"{model_name.capitalize()} job exexution is in progress --> status {constants.IN_PROGRESS_STATUS}"
                )
                return {
                    constants.MODEL_KEY: constants.MODEL_NAMES.get(model_name),
                    constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
                    constants.RESPONSE_KEY: {},
                }

            elif output_response.get(constants.STATUS_KEY) == constants.SUCCESS_KEY:
                logger.info(
                    f"{model_name.capitalize()} job exexuted successfully --> status {constants.COMPLETED_STATUS}"
                )
                return {
                    constants.MODEL_KEY: constants.MODEL_NAMES.get(model_name),
                    constants.STATUS_KEY: constants.COMPLETED_STATUS,
                    constants.RESPONSE_KEY: output_response,
                }

            elif output_response.get(constants.STATUS_KEY) == constants.ERROR_KEY:
                logger.info(
                    f"Exception encountered while serving the {model_name.capitalize()}"
                )
                logger.info(
                    f"{model_name.capitalize()} status is {constants.ERROR_STATUS}"
                )
                return {
                    constants.MODEL_KEY: constants.MODEL_NAMES.get(model_name),
                    constants.STATUS_KEY: constants.ERROR_STATUS,
                    constants.RESPONSE_KEY: {},
                }

        else:
            logger.info(f"Inference of {model_name.capitalize()} not started")
            logger.info(
                f"{model_name.capitalize()} status is {constants.JOB_NOT_FOUND_STATUS}"
            )
            return {
                constants.MODEL_KEY: constants.MODEL_NAMES.get(model_name),
                constants.STATUS_KEY: constants.JOB_NOT_FOUND_STATUS,
                constants.RESPONSE_KEY: {},
            }

    else:
        output_exist = os.path.exists(file_path)
        if output_exist:
            with open(file_path) as stream:
                output_response = json.load(stream)
                if (
                    output_response.get(constants.STATUS_KEY)
                    == constants.IN_PROGRESS_STATUS
                ):
                    logger.info(
                        f"{model_name.capitalize()} job exexution is in progress --> status {constants.IN_PROGRESS_STATUS}"
                    )
                    return {
                        constants.MODEL_KEY: constants.MODEL_NAMES.get(model_name),
                        constants.STATUS_KEY: constants.IN_PROGRESS_STATUS,
                        constants.RESPONSE_KEY: {},
                    }

                elif output_response.get(constants.STATUS_KEY) == constants.SUCCESS_KEY:
                    logger.info(
                        f"{model_name.capitalize()} job exexuted successfully --> status {constants.COMPLETED_STATUS}"
                    )
                    return {
                        constants.MODEL_KEY: constants.MODEL_NAMES.get(model_name),
                        constants.STATUS_KEY: constants.COMPLETED_STATUS,
                        constants.RESPONSE_KEY: output_response,
                    }

                elif output_response.get(constants.STATUS_KEY) == constants.ERROR_KEY:
                    logger.info(
                        f"Exception encountered while serving the {model_name.capitalize()}"
                    )
                    logger.info(
                        f"{model_name.capitalize()} status is {constants.ERROR_STATUS}"
                    )
                    return {
                        constants.MODEL_KEY: constants.MODEL_NAMES.get(model_name),
                        constants.STATUS_KEY: constants.ERROR_STATUS,
                        constants.RESPONSE_KEY: {},
                    }

        else:
            logger.info(f"Inference of {model_name.capitalize()} not started")
            logger.info(
                f"{model_name.capitalize()} status is {constants.JOB_NOT_FOUND_STATUS}"
            )
            return {
                constants.MODEL_KEY: constants.MODEL_NAMES.get(model_name),
                constants.STATUS_KEY: constants.JOB_NOT_FOUND_STATUS,
                constants.RESPONSE_KEY: {},
            }


def handle_uploaded_files(filename: str, path: str) -> None:
    """function for handle intake files"""
    with open(path, "wb+") as destination:
        for chunk in filename.chunks():
            destination.write(chunk)


def get_input_file_as_dict(file_path: str, output_exist: bool) -> dict:
    """Returns file content as a dict"""
    if settings.USE_S3:
        s3_utils = S3Utils()
        if output_exist:
            return s3_utils.load_json(load_prefix=file_path)
        else:
            return {}
    else:
        with open(file) as stream:
            input_file = json.load(stream)
            return input_file


def write_to_output_location(output_path: str, output: dict) -> None:
    """Writes the output response to corresponding output location

    Args:
        :base_path: base path of the meeting ID
        :output_location: folder name in which content should be written
        :file_name: name of the file
        :output: output response that needs to written

    """
    file_name = str(Path(output_path).name)
    if settings.USE_S3:
        s3_utils = S3Utils()
        logger.info(f"Started process to write output file for {file_name}")
        logger.info(f"Output path is {output_path}")
        s3_utils.write_json(store_prefix=output_path, content=json.dumps(output))
    else:
        logger.info(f"Started process to write output file for {file_name}")

        # create output dir if doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output path is {output_path}")

        # Writing output to corresponding file.
        with open(output_path, "w") as destination:
            destination.write(json.dumps(output, indent=4))


def get_file_base_path(file_path: str, specs_path: str, file_name: str) -> str:
    """Returns base path of a file"""
    base_path = str(file_path).replace(f"/{file_name}", "")
    if specs_path:
        base_path = os.sep.join([base_path, specs_path])

    logger.info(f"Base path for file {file_name} is {base_path}")

    return base_path


def dict_to_json(dicty):
    json_ = json.dumps(dicty, indent=None)
    return json_


def json_to_dict(json_data):
    return json.loads(json_data)


def create_error_response(status: str, model: str):
    """Creates error response dict"""
    error_response_dict = {}
    error_response_dict[constants.STATUS_KEY] = status
    error_response_dict[constants.MODEL_KEY] = model
    error_response_dict[constants.RESPONSE_KEY] = {}
    return error_response_dict


@sync_to_async
def write_to_db(response: aiohttp.client.ClientResponse, file: File, m: str) -> None:
    """utility func to update status in ML Model table"""
    MLModelStatus.objects.update_or_create(
        file=file,
        model_name=m,
        defaults={
            constants.STATUS_KEY: response.status == 200 if response else False,
        },
    )


def update_ml_execution_status_in_db(func: Callable) -> Callable:
    """decorator which updates ML model status in DB"""

    async def wrapper(*args, **kwargs):
        op = await func(*args, **kwargs)
        model_names = args[1]
        for i, m in enumerate(model_names):
            try:
                await write_to_db(op[i], args[2], m)
            except Exception as e:
                print("Exception ", getattr(e, "message", repr(e)))

    return wrapper
