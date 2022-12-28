import json
import logging
import os

import bson
import pandas as pd

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


def path_adder(path_):
    return path_


def load_csv(path, **kwargs):
    load_path = path_adder(os.path.join(config.LOAD_DIR, path))
    return pd.read_csv(load_path, **kwargs)


def save_csv(file, path, **kwargs):
    if not os.path.exists(config.OUTPUT_RUN):
        os.mkdir(config.OUTPUT_RUN)
    save_path = path_adder(os.path.join(config.OUTPUT_RUN, path))
    file.to_csv(save_path, **kwargs)


def is_model_saved(model_path):
    path = path_adder(os.path.join(config.MODEL_DIR, model_path))
    return os.path.exists(path), path


def save_model(model_path, model, tokenizer=None):
    path = os.path.join(config.MODEL_DIR, model_path)
    if not os.path.exists(config.MODEL_DIR):
        os.mkdir(config.MODEL_DIR)
    if not os.path.exists(path):
        os.mkdir(path)
    model.save_pretrained(path)
    if tokenizer:
        tokenizer.save_pretrained(path)


def dict_to_json(dicty):
    json_ = json.dumps(dicty, indent=4, cls=JSONEncoder)
    return json_


def save_json(json_data):
    path_ = path_adder(os.path.join(config.OUTPUT_RUN, config.OUT_JSON))
    with open(path_, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JSONEncoder)
    logging.debug("[INFO] JSON Saved.")


def load_json(dir_path, file_name):
    path = path_adder(os.path.join(dir_path, file_name))
    with open(path) as file:
        json_data = json.load(file)
    return json_data
