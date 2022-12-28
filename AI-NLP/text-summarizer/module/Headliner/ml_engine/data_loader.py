import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from torch import tensor
from torch.utils.data.dataset import Dataset

from .. import config
from ..utils import utils_tools

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


class HeadlinerDataset(Dataset):
    """
    HeadlinerDataset class - to load the dataset used the __getitem__ fashion supported by the Pytorch.
    The loader supports the JSON, list and the csv format for parsing the input to the network.

    Attributes:
        model: HeadlinerBackbone Model used for tokenization method
        json_data: Input JSON data for training / serving
        is_train: Whether the mode of model is training or not
    """

    def __init__(
        self, model, json_data: Optional[Dict] = None, is_train: bool = True
    ) -> None:

        super(HeadlinerDataset, self).__init__()

        self.texts = []
        self.headline = ""

        # if JSON is provided
        if json_data is not None:
            try:
                logging.debug("Loading Json file for serving...")
                self.texts = self._extract_texts_from_json(json_data=json_data)
                if is_train is True:
                    self.headline = self._extract_headline_from_json(
                        json_data=json_data
                    )
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
                json_data = utils_tools.load_json(config.TEST_JSON)
                self.texts = self._extract_texts_from_json(json_data)
                self.headline = self._extract_headline_from_json(json_data)

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
                # kwargs = {"encoding": "ISO-8859-1", "engine": "python"}
                # Loading data from CSV
                self.data = utils_tools.load_csv(train_data_file)
                self.texts = self.data[config.text_col].values
                self.headline = self.data[config.headline_col].values
            except Exception:
                logging.debug("Error loading CSV file ...", exc_info=True)
                raise

        if config.DIVIDE_TEXT is True and is_train is False:
            self.texts = utils_tools.divide_arr(
                self.texts, config.DIVIDE_N, sent_split=True
            )

        for i, text in enumerate(self.texts):
            if not isinstance(text, str):
                self.texts[i] = ""

        self.json_data = json_data
        self.model = model
        self.is_train = is_train
        self.input_length = config.INPUT_LENGTH  # Max length for input texts
        self.output_length = config.OUTPUT_LENGTH  # max length for output headlines

    def get_index_item(self, index: int) -> Tuple[bool, Dict[str, tensor]]:
        """Returns tokenized texts and labels(if mode is train) for the given index"""

        return self.__getitem__(index)

    def __getitem__(self, index):

        # Keyword Arguments (Kwargs) for input text to be used by Tokenizer
        src_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "return_tensors": "pt",
            "max_length": self.input_length,
            "truncation": True,
        }

        # Keyword Arguments (Kwargs) for target headlines to be used by Tokenizer
        tgt_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "return_tensors": "pt",
            "max_length": self.output_length,
            "truncation": True,
        }

        kwargs = {"return_tensors": "pt", "add_special_tokens": True}
        input_text = self.clean_text(
            "headline:  " + self.texts[index]
        )  # Preprocessing the input texts

        source = self.model.tokenize(
            input_text, **src_kwargs
        )  # Tokenizing the input texts
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        if self.is_train is True:
            target_ = self.clean_text(
                self.headline[index]
            )  # Preprocessing the headlines

            targets = self.model.tokenize(
                target_, **tgt_kwargs
            )  # Tokenizing the headlines

            labels = targets["input_ids"].squeeze()
            target_mask = targets["attention_mask"].squeeze()
            labels[
                labels[:] == self.model.tokenizer.pad_token_id
            ] = -100  # Padding the labels

            encodings = {
                "input_ids": source_ids,
                "attention_mask": src_mask,
                "labels": labels,
                "decoder_attention_mask": target_mask,
            }
        else:
            encodings = {
                "input_ids": source_ids,
                "attention_mask": src_mask,
                "text": self.texts[index],
            }
        return encodings

    def clean_text(self, text: str) -> str:
        """
        Base method for applying a list of methods for preprocessing of input texts

        """
        text = self._remove_extra_spaces(text)
        text = self._de_emojify(text)
        return text

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

    def _remove_extra_spaces(self, text: str) -> str:
        """

        Parameters:
            text: input text for removing extra spaces from text

        Returns:
            text with extra spaces removed
        """
        return re.sub(" +", " ", text)

    def _extract_texts_from_json(self, json_data: Dict) -> List:
        """
        Extracting text from JSON

        Parameters:
            json_data (dict): JSON for the extraction of texts

        Returns:
            List of texts extracted

        """
        combined_texts = ""
        keys_list = json_data[config.response_column]
        for key in keys_list:
            combined_texts += key[config.text_col]
        return [combined_texts]

    def _extract_headline_from_json(self, json_data: Dict) -> List:
        """
        Extracting headline from JSON

        Parameters:
            json_data (dict): JSON for the extraction of headline

        Returns:
            Extracted Overview/ Headline

        """
        return [json_data[config.headline_col]]

    def __len__(self):
        return len(self.texts)
