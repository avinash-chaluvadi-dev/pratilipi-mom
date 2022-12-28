import logging
import os
import re
import string
from typing import Dict, List, Union

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch import tensor
from torch.utils.data.dataset import Dataset

from .. import config
from ..utils import utils_tools

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

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


class BaseDataset:
    """
    BaseDataset Class for base methods and attributes that can be inherited by another dataset class

    Attributes:
        json_data (dict): input JSON for training/ evaluating/ serving process
        is_train (bool): whether the mode is training or not

    """

    def __init__(self, json_data=None, is_train=False) -> None:

        self.texts = []
        self.labels = []
        self.original_texts = []

        # if JSON is provided
        if json_data is not None:
            try:
                logging.debug("Loading Json file for serving...")
                self.texts = self._extract_texts_from_json(json_data=json_data)
                self.chunk_ids = self._extract_id_from_json(json_data=json_data)
                if is_train is True:
                    self.labels = self._extract_labels_from_json(json_data=json_data)
            except Exception:
                logging.debug("Error Loading JSON file for serving ...", exc_info=True)
                raise
        # If JSON is not provided and mode is serve or evaluation
        elif is_train is False:
            try:
                logging.debug(
                    "Loading sample JSON file for functional testing. mode - serve / eval..."
                )
                # Loading sample JSON from dataset/ folder
                data = utils_tools.load_json(config.INPUT_DIR, config.TEST_JSON)
                self.texts = self._extract_texts_from_json(data)
                self.labels = self._extract_labels_from_json(data)
                self.chunk_ids = self._extract_id_from_json(data)
                self.json_data = data
            except Exception:
                logging.debug(
                    "Error loading JSON file for functional testing ...", exc_info=True
                )
                raise
        # Loading data for training of the model
        else:
            try:
                logging.debug("Loading training CSV file ...")
                train_data_file = config.TRAINING_DATA
                kwargs = {"encoding": "ISO-8859-1", "engine": "python"}
                # Loading data from CSV
                self.data = utils_tools.load_csv(train_data_file, **kwargs)
                self.texts = self.data[config.text_col].values
                self.labels = (
                    self.data[config.label_col].replace(config.LABEL_DICT).values
                )
            except Exception:
                logging.debug("Error loading CSV file ...", exc_info=True)
                raise

        # Replacing text value other than string with empty string
        for ind, text in enumerate(self.texts):
            if not isinstance(text, str):
                self.texts[ind] = ""

        # checking if samples are available or not
        if len(self.texts) == 0:
            logging.exception("No data samples available for model...")

        if config.PREPROCESS_TEXT is True:
            self.original_texts = self.texts.copy()
            self.texts = self.preprocess_data(self.texts)

    def _extract_texts_from_json(self, json_data):
        """
        Extracting text from JSON

        Parameters:
            json_data (dict): JSON for the extraction of texts

        """
        ret_arr = []
        keys_list = json_data[config.response_column]
        for key in keys_list:
            ret_arr.append(key[config.text_col])
        return ret_arr

    def _extract_labels_from_json(self, json_data):
        """
        Extracting Labels from JSON

        Parameters:
            json_data (dict): JSON for the extraction of labels
        """
        ret_arr = []
        keys_list = json_data[config.response_column]
        label_dict = config.LABEL_DICT
        for key in keys_list:
            ret_arr.append(label_dict[key[config.allocation_col]])
        return ret_arr

    def _extract_id_from_json(self, json_data: Dict) -> List:
        """
        Extracting chunk IDs from JSON

        Parameters:
            json_data (dict): JSON for the extraction of chunk IDs

        """
        ret_arr = []
        keys_list = json_data[config.response_column]
        for key in keys_list:
            ret_arr.append(key[config.chunk_id])
        return ret_arr

    def preprocess_data(self, data: Union[str, list, np.ndarray]) -> List[str]:
        """
        Base method for applying a list of methods for preprocessing of input texts

        """
        data = self.remove_extra_spaces(data)
        data = self.convert_to_lowercase(data)
        data = self.remove_punctuations(data)
        data = self.remove_stop_words(data)
        data = self.remove_emojis(data)
        return data

    def remove_extra_spaces(self, data: Union[str, list, np.ndarray]) -> List[str]:
        if isinstance(data, str):
            data = [data]
        for ind, text in enumerate(data):
            data[ind] = re.sub(" +", " ", text)
        return data

    def remove_punctuations(self, data: Union[str, list, np.ndarray]) -> List[str]:
        """
        Removing punctuations from given texts

        Parameters:
            data: list of texts / single text

        Returns:
            list: list of texts/ text with punctuations removed

        """
        if isinstance(data, str):
            data = [data]
        punctuations_list = set(string.punctuation)
        for ind, text in enumerate(data):
            for punc in punctuations_list:
                text = text.replace(punc, "")
            data[ind] = text
        return data

    def convert_to_lowercase(self, data: Union[str, list, np.ndarray]) -> List[str]:
        """
        Converting given texts to lowercase

        Parameters:
            data: list of texts / single text

        Returns:
            list: list of texts/ text converted to lowercase

        """
        if isinstance(data, str):
            data = [data]
        for ind, text in enumerate(data):
            data[ind] = text.lower()
        return data

    def remove_stop_words(self, data: Union[str, list, np.ndarray]) -> List[str]:
        """
        Removing stop words from given texts

        Parameters:
            data: list of texts / single text

        Returns:
            list: list of texts/ text with stopwords removed

        """

        if isinstance(data, str):
            data = [data]
        stop_words = set(stopwords.words("english"))
        for ind, text in enumerate(data):
            word_tokens = word_tokenize(text)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            data[ind] = " ".join(filtered_sentence)
        return data

    def _de_emojify(self, text: str) -> str:
        """
        Removing emojis from texts

        Parameters:
            text: String

        Returns:
            string: text with emojis removed

        """
        regex_pattern = re.compile(
            pattern="["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        return regex_pattern.sub(r"", text)

    def remove_emojis(self, data: Union[str, list, np.ndarray]) -> List[str]:
        """
        Higher level method that make use of de_emojify function to remove emojis from a list if texts

        """
        if isinstance(data, str):
            data = [data]
        for ind, text in enumerate(data):
            data[ind] = self._de_emojify(text)
        return data


class TransformerModelDataset(BaseDataset, Dataset):
    """
    TransformerModelDataset class - to load the dataset used the __getitem__ fashion supported by the Pytorch.
    The loader supports the JSON, list and the csv format for parsing the input to the network.

    Attributes:
        model: TransformerForSequenceClassification or TransformerForFeatureExtraction model
        json_data: Input JSON data for training / serving
        text_list: list of texts for training / serving
        labels_list: list of labels corresponding to text_list. Mandatory to pass this if `is_train` is True
        is_train: Whether the mode of model is training or not
        is_retrain: Whether model is retrained with the new data.
    """

    def __init__(
        self,
        model,
        json_data: Dict = None,
        text_list: List = None,
        labels_list: List = None,
        is_train: bool = False,
        is_retrain: bool = False,
    ) -> None:

        # Converting list to JSON format as digestible by the data loader/ batcher
        if text_list is not None:
            json_data = utils_tools.list_to_json(text_list, labels_list)
        self.json_data = json_data

        super(TransformerModelDataset, self).__init__(
            json_data=json_data, is_train=is_train
        )
        if is_retrain is True:
            raise NotImplementedError(
                "Retraining part of the model is yet to be implemented"
            )  # Code block for retraining of model from feedback loop [TBD].
        self.is_train = is_train
        self.is_retrain = is_retrain
        self.max_length = config.MAX_LEN
        self.model = model

    def get_index_item(self, index: int) -> Dict[str, tensor]:
        """Returns tokenized texts and labels(if mode is train) for the given index"""

        return self.__getitem__(index)

    def __getitem__(self, index):
        kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "return_tensors": "pt",
            "max_length": self.max_length,
            "truncation": True,
        }

        input_sentence = self.texts[index]
        encodings = self.model.tokenize(input_sentence, **kwargs)
        encodings = {key: data.squeeze(0) for key, data in encodings.items()}
        if self.is_retrain is True or self.is_train is True:
            label = torch.tensor(
                self.labels[index]
            )  # Adding labels if the mode is training/evaluation or retraining
            encodings.update({"labels": label})
        return encodings

    def __len__(self):
        return len(self.texts)
