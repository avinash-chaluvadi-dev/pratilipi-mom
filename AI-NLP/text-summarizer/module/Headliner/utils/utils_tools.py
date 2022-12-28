import json
import logging
import os
from typing import Dict, List, Union

import bson
import numpy as np
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer

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


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bson.ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    load_path = os.path.join(config.INPUT_DIR, path)
    return pd.read_csv(load_path, **kwargs)


def save_csv(file, path, **kwargs):
    if not os.path.exists(config.OUTPUT_DIR):
        os.mkdir(config.OUTPUT_DIR)
    save_path = os.path.join(config.OUTPUT_DIR, path)
    file.to_csv(save_path, **kwargs)


def is_model_saved(model_path):
    path = os.path.join(config.MODEL_DIR, model_path)
    return os.path.exists(path), path


def save_model(
    model_path: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer = None
) -> None:
    if model_path == config.FINETUNED_MODEL:
        is_saved, path = is_model_saved(config.BASE_FINETUNED_MODEL)
        if not is_saved:
            model_path = config.BASE_FINETUNED_MODEL
    path = os.path.join(config.MODEL_DIR, model_path)
    if not os.path.exists(config.MODEL_DIR):
        os.mkdir(config.MODEL_DIR)
    if not os.path.exists(path):
        os.mkdir(path)
    model.save_pretrained(path)
    if tokenizer:
        tokenizer.save_pretrained(path)
    logging.info(f"{model_path} Model Saved.")


def dict_to_json(dicty):
    json_ = json.dumps(dicty, indent=4, cls=JSONEncoder)
    return json_


def json_to_dict(json_data):
    return json.loads(json_data)


def save_json(json_data, json_path=None):
    if json_path is None:
        json_path = config.OUT_JSON
    path_ = os.path.join(config.OUTPUT_RESULTS, json_path)
    with open(path_, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JSONEncoder)
    logging.info(f"JSON Saved.")


def load_json(json_path=None):
    if json_path is None:
        json_path = config.TEST_JSON
    path = os.path.join(config.INPUT_DIR, json_path)
    with open(path) as file:
        json_data = json.load(file)
    return json_data


def parse_json(json_data):
    return json.loads(json.dumps(json_data, cls=JSONEncoder))


def divide_arr(arr, num_chunk, sent_split=False) -> List:
    """
    Function to divide the array into small chunks
    Parameters:
        arr: Array to be divided
        num_chunk: num of splits to be made in array
        sent_split: if passed array contains one element with string

    Returns:

    """
    if sent_split is True:
        split_arr = arr[0].split(".")
    else:
        split_arr = arr[:]

    for ind, txt in enumerate(split_arr):
        if not isinstance(txt, str):
            split_arr[ind] = ""

    ret_arr = []
    arr_len = len(split_arr)
    part = arr_len // num_chunk
    start = 0
    for _ in range(num_chunk):
        string = ".".join(split_arr[start : start + part])
        ret_arr.append(string)
        start += part
    return ret_arr


def get_response(prediction: str, json_data: Dict, status: str) -> Dict:
    """

    Parameters:
        prediction: headline (overview) prediction by the headliner model
        json_data: Input JSON
        status: Success or fail/ failure

    Returns:
        Response JSON
    """
    status = status.lower()
    json_data["status"] = status
    json_data["model"] = "headliner"
    if status not in ["fail", "failure", "error"]:
        json_data[config.headline_col] = prediction
    return json_data


def save_prediction(text_list: list, predictions: Union[list, np.ndarray]) -> None:
    """
    Function to save predictions locally (used for functional testing)

    Parameters:
        text_list: list of texts
        predictions: list of predicted headlines

    """
    if len(text_list) != len(predictions):
        logging.exception("Length of predictions and texts are not same.")

    result = dict(zip(text_list, predictions))
    save_json(result)  # saving the predictions locally
