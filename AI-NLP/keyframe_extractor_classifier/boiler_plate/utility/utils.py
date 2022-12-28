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
import sys
import time
from pathlib import Path

import jwt

logger = lg.getLogger("file")
import json

import boto3
import bson
import pandas as pd
from botocore import exceptions
from django.conf import settings
from glob2 import glob

from boiler_plate.utility import constants


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


def handle_uploaded_files(filename, path):
    """function for handle intake files"""
    with open(path, "wb+") as destination:
        for chunk in filename.chunks():
            destination.write(chunk)


def get_input_file_as_dict(file_path: str, output_exist: bool = True) -> dict:
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


def create_error_json(model_name):
    return {"status": "success", "model": model_name, "response": []}


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
    base_path = str(file_path).replace(file_name, "")
    if specs_path:
        base_path = f"{base_path}/{specs_path}"
    logger.info(f"Base path for file {file_name} is {base_path}")
    return base_path


def create_error_response(status: str, model: str):
    """Creates error response dict"""
    error_response_dict = {}
    error_response_dict[constants.STATUS_KEY] = status
    error_response_dict[constants.MODEL_KEY] = model
    error_response_dict[constants.RESPONSE_KEY] = {}
    return error_response_dict


def set_sessiontransection_dtls(sesobject, runtimedictionary, sestype):
    if sesobject:
        sesobject["customuserid"] = runtimedictionary["id"]
        sesobject["domainid"] = runtimedictionary["domainid"]
        sesobject["transectionlock"] = True
        sesobject["authorize_timestamp"] = runtimedictionary["authorize_timestamp"]
        if "block" in sestype:
            sesobject["actionitem"] = "user working action item"
            sesobject["comments"] = "user previous token blocked"
        else:
            sesobject["actionitem"] = "user working action item"
            sesobject["comments"] = "user working action item"
        sesobject["modified"] = str(time.strftime("%Y-%m-%d"))
    return sesobject


def set_session_prohibition_dtls(sesobject, runtimedictionary):
    if sesobject:
        sesobject["customuserid"] = runtimedictionary["id"]
        sesobject["domainid"] = runtimedictionary["domainid"]
        sesobject["email"] = runtimedictionary["email"]
        sesobject["authorize_timestamp"] = runtimedictionary["authorize_timestamp"]
        sesobject["comments"] = "session logout successfully"
        sesobject["modified"] = str(time.strftime("%Y-%m-%d"))
        sesobject["is_blocked_status"] = True
    return sesobject


def get_sestokens(email, sessecret):
    try:
        if email != None:
            pra_encoded_jwt_access_key = jwt.encode(
                {
                    "email": email,
                },
                sessecret,
                algorithm="HS256",
            )
        return pra_encoded_jwt_access_key
    except Exception as error:
        print("Exception ::", error)


def dict_to_json(dicty):
    json_ = json.dumps(dicty, indent=None, cls=JSONEncoder)
    return json_


def json_to_dict(json_data):
    return json.loads(json_data)


def get_request_datasets(foldername, stime, runtime_uuid, userId):
    """:---: Get all result set path :---:"""
    json_dir_name = "{0}".format(foldername)
    filter_pattern = os.path.join(json_dir_name, f"{json_dir_name}/**/*.json")
    file_list = glob(filter_pattern)
    dataSet = {}
    for file in file_list:
        indexval = file.split(".")[0].split("\\")
        if "mom" in indexval:
            print(
                "----<:file",
                file,
                "---<:uuid",
                runtime_uuid,
                "----<:userid",
                userId,
            )
            with open(f"{file}") as result_set:
                dataSet[runtime_uuid] = json.loads(
                    json.loads(dict_to_json(result_set.read()))
                )

    """
    with open("{0}_consolidate_dataset.json".format(stime), "a") as writer:
        writer.write(json.dumps(dataSet))
    """
    return dataSet


def check_pratilipi_reqids(uuid_to_test, version=4):
    from uuid import UUID

    """
    Check if uuid_to_test is a valid UUID.
    uuid_to_test : str
    version : {1, 2, 3, 4}
    returns: true/false
    """
    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False

    return str(uuid_obj) == uuid_to_test


def get_outputpath_reqids(output_syspath):
    dirdict = []
    try:
        for dir_uuid_to_test in os.listdir(output_syspath):
            if check_pratilipi_reqids(dir_uuid_to_test, version=4):
                dirdict.append(dir_uuid_to_test)
    except ValueError:
        return False

    return dirdict


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bson.ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def get_marker_lable(*args):
    try:
        val = args[0]
        if val >= 999:
            return "Exceptional"
        elif val >= 500:
            return "Good"
        elif val >= 300:
            return "Fair"
        else:
            return "Poor"

    except Exception as err:
        print("Exception::", err)
