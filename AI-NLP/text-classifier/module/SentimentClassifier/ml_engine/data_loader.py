import logging
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import config
from ..utils import utils_tools
from .model import SentimentBackbone

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


def clean_text(text):
    # text = text.replace("\n", "").replace("`", "").replace('"', '')
    return text


def sentiments_mapping(sentiments):
    sentiments = pd.Series(sentiments)
    return sentiments.map({"neutral": 1, "negative": 0, "positive": 2}).values


def extract_texts_from_json(json_data, column_name):
    """
    Extracting text from JSON

    """
    ret_arr = []
    keys_list = json_data[column_name]
    for ind, key in enumerate(keys_list):
        ret_arr.append(key[f"{config.TextColumn}"])
    return ret_arr


def extract_labels_from_json(json_data, column_name):
    """
    Extracting Labels from JSON

    """
    ret_arr = []
    keys_list = json_data[column_name]
    label_dict = {"neutral": 1, "negative": 0, "positive": 2}
    for ind, key in enumerate(keys_list):
        ret_arr.append(label_dict[key[config.ClassifierColumn]])
    return ret_arr


class ClassifierDataset(Dataset):
    """
    Classifier-Dataset class - to load the dataset used the __getitem__ fashion supported by the Pytorch.
    The loader supports the JSON forma for the training of the model.

    """

    def __init__(self, model: SentimentBackbone, load_mode="serve", json_data=None):
        super(ClassifierDataset, self).__init__()

        # Training Mode / Evaluation Mode Data Loading.
        if load_mode == "train" or load_mode == "eval":
            try:
                logging.debug("Loading training JSON file ...")
                data = utils_tools.load_json(config.LOAD_DIR, config.TRAIN_JSON)
                self.texts = extract_texts_from_json(
                    data, config.TRAIN_TranscriptionColumn
                )
                self.labels = extract_labels_from_json(
                    data, config.TRAIN_TranscriptionColumn
                )
            except Exception as e:
                logging.exception("Error loading the training JSON", exc_info=True)
                raise

        # Serving mode Data Loading.
        elif load_mode == "serve":
            try:
                logging.debug(
                    "Loading JSON for the functional test run of Serving mode ..."
                )
                # In the serving mode, the input is passed instead of loading it from a source, due to the
                # sequential orchestration of the N models.
                data = json_data
                self.texts = extract_texts_from_json(
                    data, config.SERVE_TranscriptionColumn
                )
                self.is_train = False
            except Exception:
                logging.exception(
                    "Error loading the json file for the functional test mode - Serving",
                    exc_info=True,
                )
                raise

        else:
            raise logging.exception("Invalid Data loading Mode ...", exc_info=True)

        for i, text in enumerate(self.texts):
            if not isinstance(text, str):
                self.texts[i] = ""

        if load_mode == "train":
            self.is_train = True
        self.model = model

    def __getitem__(self, index):
        src_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
        }

        input_ = clean_text(self.texts[index])
        source = self.model.tokenize(input_, **src_kwargs)
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        if self.is_train:
            labels = torch.tensor(self.labels[index])
            return {
                "input_ids": source_ids,
                "attention_mask": src_mask,
                "labels": labels,
            }
        else:
            return {
                "input_text": input_,
                "input_ids": source_ids,
                "attention_mask": src_mask,
            }

    def __len__(self):
        return len(self.texts)
