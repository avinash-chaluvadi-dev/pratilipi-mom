import json
import logging
import os
import shutil
from typing import Dict, List, Optional

import boto3
import bson
import imagehash
import numpy as np
from botocore import exceptions
from botocore.client import Config
from PIL import Image, ImageFilter

from .. import config

if not config.USE_EFS:
    logging.basicConfig(
        filename=os.path.join(config.OUTPUT_LOG, config.LOG_FILE),
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )


class S3Utils:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3", config=Config(signature_version="s3v4"))

    def load_json(self, load_prefix):
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=load_prefix)
            return json.loads(obj["Body"].read().decode("utf-8"))

        except exceptions.DataNotFoundError as error:
            logging.error("Specified prefix does not exist")
            raise RuntimeError(error)

    def put_data(self, store_prefix: str, content: str, content_type=None) -> None:
        try:
            if not content_type:
                # Put the data into store_path of corresponding S3 bucket
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=store_prefix,
                    Body=content,
                )
            else:
                # Put the data into store_path of corresponding S3 bucket
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=store_prefix,
                    Body=content,
                    ContentType=content_type,
                )

        except exceptions.ClientError as error:
            logging.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def prefix_exist(self, file_prefix):
        """Function to check whether prefix exist of not"""
        try:
            # Creates bucket object from s3 resource
            bucket = self.s3_resource.Bucket(self.bucket_name)
            logging.info(f"Started checking whether {file_prefix} exist or not")
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
            logging.error(error.response["Error"]["Message"])
            raise RuntimeError(error)

    def parse_s3_uri(self, file_uri: str):
        _, base_prefix = file_uri.replace(config.S3_URI, "").split("/", 1)
        return base_prefix

    def generate_presigned_uri(self, object_prefix: str, uri_timeout=None):
        try:
            s3_source_signed_uri = self.s3_resource.meta.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_prefix},
                ExpiresIn=6000,
            )
            return s3_source_signed_uri
        except exceptions.ClientError as error:
            logging.error(error.response["Error"]["Message"])
            raise RuntimeError(error)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bson.ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def hashing_difference(img1: np.ndarray, img2: np.ndarray):
    """
    Utility function to return hashing difference between two images.
    Returns three type of hashing difference - aHash, pHash, dHash
    """
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    if img1.width < img2.width:
        img2 = img2.resize((img1.width, img1.height))
    else:
        img1 = img1.resize((img2.width, img2.height))
    img1 = img1.filter(ImageFilter.BoxBlur(radius=3))
    img2 = img2.filter(ImageFilter.BoxBlur(radius=3))

    phashvalue = imagehash.phash(img1) - imagehash.phash(img2)
    ahashvalue = imagehash.average_hash(img1) - imagehash.average_hash(img2)
    dhashvalue = imagehash.dhash_vertical(img1) - imagehash.dhash_vertical(img2)
    return phashvalue, ahashvalue, dhashvalue


def get_hash(img: np.ndarray):
    """
    Utility function to find image hash of an image
    """
    img = Image.fromarray(img)
    img = img.filter(ImageFilter.BoxBlur(radius=3))
    return (
        imagehash.phash(img),
        imagehash.average_hash(img),
        imagehash.dhash_vertical(img),
    )


def precision_score(true_positive: float, false_positive: float) -> float:
    if true_positive + false_positive == 0:
        logging.warning(
            "True Positive and False positive sums to 0. Can not find Precision score..."
        )
        return 0
    return true_positive / (true_positive + false_positive)


def recall_score(true_positive: float, false_negative: float) -> float:
    if true_positive + false_negative == 0:
        logging.warning(
            "True Positive and False negative sums to 0. Can not find Recall score..."
        )
        return 0
    return true_positive / (true_positive + false_negative)


def f1_score(
    true_positive: float, false_positive: float, false_negative: float
) -> float:
    precision = precision_score(true_positive, false_positive)
    recall = recall_score(true_positive, false_negative)
    if precision + recall == 0:
        logging.warning("Precision and Recall sums to 0. Can not find F1 score...")
        return 0
    return 2 * precision * recall / (precision + recall)


def dict_to_json(dicty):
    json_ = json.dumps(dicty, indent=4, cls=JSONEncoder)
    return json_


def json_to_dict(json_data):
    return json.loads(json_data)


def save_json(json_data):
    path_ = os.path.join(config.OUTPUT_RESULTS, config.OUT_JSON)
    with open(path_, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JSONEncoder)
    logging.info(f"JSON Saved.")


def load_json(json_path=None):
    if json_path is None:
        json_path = config.TEST_JSON
    if config.USE_S3:
        s3_utils = S3Utils(bucket_name=config.AWS_STORAGE_BUCKET_NAME)
        json_data = s3_utils.load_json(load_prefix=json_path)
    else:
        path = os.path.join(config.INPUT_DIR, json_path)
        with open(path) as file:
            json_data = json.load(file)
    return json_data


def parse_json(json_data):
    return json.loads(json.dumps(json_data, cls=JSONEncoder))


def empty_dir(path) -> None:
    """
    Utility function to clear a directory
    """
    if os.path.exists(path):
        for content in os.listdir(path):
            content_path = os.path.join(path, content)
            if os.path.isfile(content_path):
                os.remove(content_path)
            elif os.path.isdir(content_path):
                shutil.rmtree(os.path.join(path, content))
            elif os.path.islink(content_path):
                os.unlink(content_path)
    return None


def get_response(
    keyframe_paths: Optional[List[list]], json_data: Dict, status: str
) -> Dict:
    """

    Parameters:
        keyframe_paths: list of keyframe paths for each chunk
        json_data: Input JSON
        status: Success or fail/ failure

    Returns:
        Response JSON
    """

    json_data["status"] = status.lower()
    json_data["model"] = "Framify (Keyframe Extractor)"
    status = status.lower()
    for index, chunk_dict in enumerate(json_data[config.response_key]):
        paths = keyframe_paths[index]
        confidence_score = None
        if status in ["error", "fail", "failure"]:
            paths = []
            confidence_score = 0.00
        chunk_dict[config.keyframe_key] = paths
        chunk_dict["confidence_score"] = confidence_score
        json_data[config.response_key][index] = chunk_dict
    return json_data


def save_prediction(response_dict: Dict) -> None:
    """
    Function to save response dictionary locally (used for functional testing)

    Parameters:
        response_dict: Response dictionary to be saved locally

    """
    save_json(response_dict)
