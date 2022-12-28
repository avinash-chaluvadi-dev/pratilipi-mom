import json
import logging
import os
from typing import Dict, List, Optional, Union

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


def list_to_json(
    text_list: List, labels_list: Optional[List] = None
) -> Dict[str, list]:
    """
    Function that converts a list of texts to defined JSON format for passing it to the dataset class

    Parameters:
        text_list: list of texts (str)
        labels_list: list of labels corresponding to texts (mandatory if mode is train)
    Returns:
        JSON data in the required format

    """
    json_data = dict()
    json_data[config.response_column] = []
    for ind, text in enumerate(text_list):
        transcript = dict()
        transcript[config.text_col] = text
        transcript[config.chunk_id] = ind + 1
        if labels_list is not None:
            transcript[config.summary_col] = labels_list[ind]
        json_data[config.response_column].append(transcript)
    return json_data


def get_response(predictions: List, json_data: Dict, status: str) -> Dict:
    """

    Parameters:
        predictions: list of predictions/ summaries
        json_data: Input JSON
        status: Success or fail/ failure

    Returns:
        Response JSON
    """
    status = status.lower()
    json_data["status"] = status
    json_data["model"] = "summarizer"

    for index, chunk_dict in enumerate(json_data[config.response_column]):
        prediction = predictions[index]
        confidence_score = None
        if status in ["fail", "failure", "error"]:
            prediction = None
            confidence_score = 0.0
        chunk_dict[config.summary_col] = prediction
        chunk_dict["confidence_score"] = confidence_score
        json_data[config.response_column][index] = chunk_dict
    return json_data


def save_prediction(text_list: list, predictions: Union[list, np.ndarray]):
    """
    Function to save predictions locally (used for functional testing)

    Parameters:
        text_list: list of texts
        predictions: list of predicted summaries

    """
    if len(text_list) != len(predictions):
        logging.exception("Length of predictions and texts are not same.")

    result = dict(zip(text_list, predictions))
    save_json(result)  # saving the predictions locally
