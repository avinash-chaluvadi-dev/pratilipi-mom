import json
import logging
import os
from json.encoder import JSONEncoder

from .. import config


def path_exists(path):
    return os.path.exists(path)


def save_model(nlp):
    model_dir = os.path.join(config.MODELS_DIR, config.MODEL_NAME_SAVE)
    if not path_exists(model_dir):
        os.makedirs(model_dir)
    nlp.to_disk(model_dir)
    logging.debug(f"Saved model to {model_dir}")


def save_result(json_data):
    """
    Saves response to the OUT.JSON file in results directory.
    """
    store_path = os.path.join(config.OUTPUT_RUN, config.OUT_JSON)
    if not os.path.exists(config.OUTPUT_RUN):
        os.makedirs(config.OUTPUT_RUN)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    logging.debug("[INFO] JSON Saved.")


def save_json(json_data):
    path_ = os.path.join(config.OUTPUT_RUN, config.OUT_JSON)
    with open(path_, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JSONEncoder)
    logging.debug("[INFO] JSON Saved.")


def load_json(dir_path, file_name):
    path = os.path.join(dir_path, file_name)
    with open(path) as file:
        json_data = json.load(file)
    return json_data
