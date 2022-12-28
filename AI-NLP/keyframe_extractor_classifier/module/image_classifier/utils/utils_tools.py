import json
import os

import boto3
import torch
from botocore import exceptions

from .. import config


def path_exists(path):
    return os.path.exists(path)


class S3Utils:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")

    def load_data(self, load_prefix):
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=load_prefix)
            return obj["Body"].read()

        except exceptions.DataNotFoundError as error:
            logging.error("Specified prefix does not exist")
            raise RuntimeError(error)

    def put_data(self, store_prefix: str, content: str) -> None:
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=store_prefix, Body=content
            )

        except exceptions.ClientError as error:
            logging.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def parse_s3_uri(self, file_uri: str):
        _, base_prefix = file_uri.replace(config.S3_URI, "").split("/", 1)
        return base_prefix


def create_output_json(status, json_data, response):
    json_data["status"] = status.lower()
    json_data["model"] = "keyframe_classifier"
    chunk_ids = response.get("chunk_ids")
    predictions = response.get("predictions")
    confidence_score = response.get("confidence_score")
    for keyframe in json_data.get(config.params.get("infer").get("response_key")):
        if len(keyframe.get("keyframes")) == 0:
            keyframe["keyframe_labels"] = []
            keyframe["confidence_score"] = []
        if keyframe.get("speaker_label") in chunk_ids:
            start_index = chunk_ids.index(keyframe.get("speaker_label"))
            end_index = (
                len(chunk_ids)
                - chunk_ids[::-1].index(keyframe.get("speaker_label"))
                - 1
            )
            keyframe["keyframe_labels"] = predictions[start_index : end_index + 1]
            keyframe["confidence_score"] = confidence_score[start_index : end_index + 1]
    return json_data


def save_result(json_data):
    """
    Saves response to the OUT.JSON file in results directory.
    """
    store_path = os.path.join(config.OUTPUT_RUN, config.OUT_JSON)
    if not os.path.exists(config.OUTPUT_RUN):
        os.makedirs(config.OUTPUT_RUN)
    with open(store_path, "w") as f:
        json.dump(json_data, f)


def save_model(model_name, model):
    model_dir = config.MODELS_DIR
    if not path_exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
