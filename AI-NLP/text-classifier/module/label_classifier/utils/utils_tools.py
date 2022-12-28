import json
import os
from os.path import dirname as up

import pandas as pd
import torch
import transformers

from .. import config


def create_dataframe(json_data):
    """
    Creates a data frame from Json.
    """
    text_list = json_data.get(config.params.get("json").get("text_key"))
    label_list = json_data.get(config.params.get("json").get("label_key"))
    return pd.DataFrame({"text": text_list, "labels": label_list})


def get_specs(logger, key):
    """
    Returns value from 'specs.json' corresponding to input key
        :param logger: custom_logger to log information
        :param key: key to get value
        :return: value from 'specs.json' corresponding to key

    """
    json_path = os.path.join(up(up(os.path.abspath(__file__))), "specs.json")
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
            return json_data.get(key)
    except (FileNotFoundError, RuntimeError):
        logger.exception("'specs.json' file is not found..")


def modify_specs(specs):
    """

    Returns value from 'specs.json' corresponding to input key
        :param specs: To modify/update the dictionary present in 'specs.json'

    """
    json_path = os.path.join(up(up(os.path.abspath(__file__))), "specs.json")
    try:
        with open(json_path, "r+") as f1:
            json_data = json.load(f1)
            with open(json_path, "w") as f2:
                for key, value in specs.items():
                    json_data[key] = value
                json.dump(json_data, f2)
    except FileNotFoundError:
        with open(json_path, "w") as f:
            json.dump(specs, f)


def path_exists(path):
    """

    To check whether path exists or not
        :param path: Input path string
        :return bool: Returns True if path exists otherwise returns False

    """
    return os.path.exists(path)


def is_model_saved(model_path):
    """

    To check whether model is saved or not
        :param model_path: model_path string
        :return bool: Returns True if model is saved otherwise returns False

    """
    bert_path = os.path.join(config.MODELS_DIR, model_path)
    return path_exists(bert_path)


def get_model_path(logger):
    """

    Returns the model_path(can be pretrained or finetuned path) from 'specs.json' file
        :param logger: custom_logger object to log the information
        :return: path of model_checkpoint

    """
    try:
        json_path = os.path.join(up(up(os.path.abspath(__file__))), "specs.json")
        with open(json_path, "r") as f:
            json_data = json.load(f)
            if json_data.get("model"):
                model_path = json_data.get("model")
                return model_path
            else:
                logger.info("Model path has to be defined in the 'specs.json' file..")
                return None

    except (FileNotFoundError, RuntimeError):
        logger.info("'specs.json' file is not found..")
        if is_model_saved(config.BERT_MODEL):
            model_path = config.BERT_MODEL
            return model_path
        else:
            return None


def get_model_and_tokenizer(logger, checkpoint=None):
    """

    Returns model checkpoint and tokenizer
        :param logger: custom_logger object to log the information
        :param checkpoint: Name of the checkpoint used to restore the model
        :return: model and tokenizer

    """

    if checkpoint:
        checkpoint_path = os.path.join(config.MODELS_DIR, checkpoint)
        if is_model_saved(model_path=checkpoint_path):
            model = transformers.BertModel.from_pretrained(
                os.path.join(config.MODELS_DIR, checkpoint)
            )
            tokenizer = transformers.BertTokenizer.from_pretrained(
                os.path.join(config.MODELS_DIR, checkpoint)
            )
            return model, tokenizer
        else:
            model = transformers.BertModel.from_pretrained(checkpoint)
            tokenizer = transformers.BertTokenizer.from_pretrained(checkpoint)
            return model, tokenizer

    else:
        model_path = get_model_path(logger=logger)
        if model_path and is_model_saved(model_path=model_path):
            try:
                bert_path = os.path.join(config.MODELS_DIR, model_path)
                model = transformers.BertModel.from_pretrained(bert_path)
                tokenizer = transformers.BertTokenizer.from_pretrained(bert_path)
            except (RuntimeError, ValueError):
                logger.info("Downloading Model...")
                model = transformers.BertModel.from_pretrained(model_path)
                tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
                save_model(
                    logger=logger,
                    model_name=config.BERT_MODEL,
                    model=model,
                    tokenizer=tokenizer,
                )
            return model, tokenizer
        else:
            logger.info("Downloading Model...")
            model = transformers.BertModel.from_pretrained(config.BERT_MODEL)
            tokenizer = transformers.BertTokenizer.from_pretrained(config.BERT_MODEL)
            save_model(
                logger=logger,
                model_name=config.BERT_MODEL,
                model=model,
                tokenizer=tokenizer,
            )
            modify_specs(specs={"model": config.BERT_MODEL})
            return model, tokenizer


def save_result(json_data):
    """

    Saves response to the OUT.JSON file in results directory.
        :param json_data: Input json to save into OUT.JSON file

    """
    store_path = os.path.join(config.OUTPUT_RUN, config.OUT_JSON)
    if not os.path.exists(config.OUTPUT_RUN):
        os.makedirs(config.OUTPUT_RUN)
    with open(store_path, "w") as f:
        json.dump(json_data, f)


def load_state_dict(logger, model):
    """
    Loads the state_dict(weights/parameters) into model object after finetuning
        :param logger: custom_logger object to log the information
        :param model: Name of the checkpoint used to load the state_dict

    """
    model.load_state_dict(
        torch.load(
            os.path.join(
                config.MODELS_DIR,
                get_specs(logger=logger, key="model"),
                config.LABEL_CLASSIFIER_BIN,
            ),
            map_location="cpu",
        )
    )


def save_model(
    logger,
    model_name,
    model,
    tokenizer=None,
):
    """

    Saves the model bin file and tokenizer vocabulary file
        :param logger: custom_logger object to log the information
        :param model_name: Name of the checkpoint used to save the model
        :param model: model checkpoint used to save the state_dict of model object
        :param tokenizer: tokenizer used to save the state

    """
    model_path = os.path.join(config.MODELS_DIR, model_name)
    if not path_exists(model_path):
        os.makedirs(model_path)
    if hasattr(model, "model"):
        model.model.save_pretrained(model_path)
        torch.save(
            model.state_dict(),
            os.path.join(model_path, config.LABEL_CLASSIFIER_BIN),
        )
    else:
        model.save_pretrained(model_path)
    logger.info("Model has been successfully saved..")
    if tokenizer:
        tokenizer.save_pretrained(model_path)


def create_response_dict(response):
    """
    Creates a dictionary with key as text and value as tuple containing prediction and confidence_score
        :param response: Marker Classifier response
        :return: dictionary with key as text and value as tuple containing prediction and confidence_score
    """
    response_dict = {}
    for text, prediction, confidence_score in zip(
        response.get("text"),
        response.get("predictions"),
        response.get("confidence_score"),
    ):
        response_dict[text] = (prediction, confidence_score)
    return response_dict


def create_output_json(status, json_data, response):
    """
    Creates output json response
        :param json_data: Input json to update with model predictions
        :param response: Marker Classifier response
        :return: json response with Label Classification predictions
    """
    json_data["status"] = status.lower()
    json_data["model"] = "label_classifier"
    response_dict = create_response_dict(response)
    for diarized_frame in json_data.get(config.params.get("json").get("response_key")):
        text = " ".join(
            diarized_frame.get(config.params.get("json").get("text_key")).split()
        )
        if text in response_dict:
            diarized_frame["label"] = response_dict.get(text)[0]
            diarized_frame["confidence_score"] = response_dict.get(text)[1]
    return json_data
