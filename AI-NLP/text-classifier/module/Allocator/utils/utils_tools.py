import json
import logging
import os
import pickle
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
    def default(self, obj):
        if isinstance(obj, bson.ObjectId):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def is_model_saved(model_path: str) -> tuple:
    path = os.path.join(config.MODEL_DIR, model_path)
    return os.path.exists(path), path


def save_model(
    model_path: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer = None
) -> None:
    if model_path == config.FINETUNED_CLASSIFIER:
        is_saved, path = is_model_saved(config.BASE_FINETUNED_CLASSIFIER)
        if not is_saved:
            model_path = config.BASE_FINETUNED_CLASSIFIER
    path = os.path.join(config.MODEL_DIR, model_path)
    if not os.path.exists(config.MODEL_DIR):
        os.mkdir(config.MODEL_DIR)
    if not os.path.exists(path):
        os.mkdir(path)
    model.save_pretrained(path)
    if tokenizer:
        tokenizer.save_pretrained(path)
    logging.info(f"{model_path} Model Saved.")


def load_csv(path, **kwargs):
    load_path = os.path.join(config.INPUT_DIR, path)
    return pd.read_csv(load_path, **kwargs)


def load_json(dir_path, file_name):
    path = os.path.join(dir_path, file_name)
    with open(path) as file:
        json_data = json.load(file)
    return json_data


def dict_to_json(data_dict):
    json_ = json.dumps(data_dict, indent=4, cls=JSONEncoder)
    return json_


def save_json(json_data, json_path=None):
    if json_path is None:
        json_path = config.OUT_JSON
    path_ = os.path.join(config.OUTPUT_RESULTS, json_path)
    with open(path_, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JSONEncoder)
    logging.info("JSON Saved.")


def parse_json(json_data):
    return json.loads(json.dumps(json_data, cls=JSONEncoder))


def pickle_dump(model, model_name):
    is_saved, path = is_model_saved(config.BASE_ML_MODEL)
    if not is_saved:
        model_name = config.BASE_ML_MODEL
    path = os.path.join(config.MODEL_DIR, model_name)
    with open(path, "wb") as file:
        pickle.dump(model, file)
    logging.info(f"[INFO] {model_name} Saved.")


def pickle_load(model_name):
    path = os.path.join(config.MODEL_DIR, model_name)
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model


def list_to_json(text_list: List, labels_list: List = None) -> Dict[str, list]:
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
            transcript[config.allocation_col] = config.INVERSE_LABEL_DICT[
                labels_list[ind]
            ]
        json_data[config.response_column].append(transcript)
    return json_data


def softmax(array_: Union[list, np.ndarray]):
    """Compute softmax values for each sets of scores in array."""
    array_ = np.array(array_)
    e_x = np.exp(array_ - np.max(array_))
    return e_x / e_x.sum(axis=0)  # only difference


def get_response(
    predictions: List, confidence_scores: List, json_data: Dict, status: str
) -> Dict:
    """

    Parameters:
        predictions: list of predictions
        confidence_scores: list of confidence scores
        json_data: Input JSON
        status: Success or fail/ failure

    Returns:
        Response JSON
    """
    json_data["status"] = status
    json_data["model"] = "Allocator (Deadline and Escalation Classifier)"
    status = status.lower()

    for index, chunk_dict in enumerate(json_data[config.response_column]):
        prediction = predictions[index]
        confidence_score = confidence_scores[index]
        if status in ["fail", "failure", "error"]:
            prediction = None
            confidence_score = 0.0
        chunk_dict["classifier_output"] = prediction
        chunk_dict["confidence_score"] = confidence_score
        json_data[config.response_column][index] = chunk_dict

    return json_data


def save_prediction(text_list: List, predictions: List) -> None:
    """
    Function to save predictions locally (used for functional testing)

    Parameters:
        text_list: list of texts
        predictions: list of prediction labels

    """
    if len(text_list) != len(predictions):
        logging.exception("Length of predictions and texts are not same.")

    result = dict(zip(text_list, predictions))
    save_json(result)  # saving the predictions locally
