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


class SummarizerDataset(Dataset):
    """
    SummarizerDataset class - to load the dataset used the __getitem__ fashion supported by the Pytorch.
    The loader supports the JSON, list and the csv format for parsing the input to the network.

    Attributes:
        model: SummarizerBackbone Model used for tokenization method
        json_data: Input JSON data for training / serving
        text_list: list of texts for training / serving
        summary_list: list of summaries corresponding to text_list. Mandatory to pass this if `is_train` is True
        is_train: Whether the mode of model is training or not
        retrain: Whether model is retrained with the new data.
    """

    def __init__(
        self,
        model,
        json_data: Optional[Dict] = None,
        text_list: Optional[List] = None,
        summary_list: Optional[List] = None,
        is_train: bool = True,
        retrain: bool = False,
    ) -> None:
        if text_list is not None:
            json_data = utils_tools.list_to_json(text_list, summary_list)

        super(SummarizerDataset, self).__init__()
        if retrain is True:
            # Code block for retraining of model from feedback loop [TBD].
            raise NotImplementedError(
                "Retraining part of the model is yet to be implemented"
            )

        self.texts = []
        self.summaries = []

        # if JSON is provided
        if json_data is not None:
            try:
                logging.debug("Loading Json file for serving...")
                self.texts = self._extract_texts_from_json(json_data=json_data)
                if is_train is True:
                    self.summaries = self._extract_summaries_from_json(
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
                self.summaries = self._extract_summaries_from_json(json_data)
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
                self.data = utils_tools.load_csv(train_data_file)
                self.texts = self.data[config.text_col].values
                self.summaries = self.data[config.summary_col].values
            except Exception:
                logging.debug("Error loading CSV file ...", exc_info=True)
                raise

        # checking if samples are available or not
        if len(self.texts) == 0:
            logging.exception("No data samples available...")

        self.json_data = json_data
        self.is_train = is_train
        self.retrain = retrain
        self.input_length = config.INPUT_LENGTH  # Max length for input texts
        self.output_length = config.OUTPUT_LENGTH  # max length for output summaries
        self.model = model

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

        # Keyword Arguments (Kwargs) for target summaries to be used by Tokenizer
        tgt_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "return_tensors": "pt",
            "max_length": self.output_length,
            "truncation": True,
        }

        text = self.texts[index]
        input_ = self.clean_text("summarize:  " + text)  # Preprocessing the input texts

        source = self.model.tokenize(input_, **src_kwargs)  # Tokenizing the input texts
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        labels = []
        target_mask = []

        if self.is_train is True or self.retrain is True:
            # Preprocessing and tokenizing the summaries when mode is training

            target_ = self.clean_text(
                self.summaries[index]
            )  # Preprocessing the summaries
            targets = self.model.tokenize(
                target_, **tgt_kwargs
            )  # Tokenizing the summaries

            labels = targets["input_ids"].squeeze()
            target_mask = targets["attention_mask"].squeeze()
            labels[
                labels[:] == self.model.tokenizer.pad_token_id
            ] = -100  # Padding the labels

        if self.is_train is True or self.retrain is True:
            # If mode is training
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
        ret_arr = []
        keys_list = json_data[config.response_column]
        for key in keys_list:
            ret_arr.append(key[config.text_col])
        return ret_arr

    def _extract_summaries_from_json(self, json_data: Dict) -> List:
        """
        Extracting Summaries from JSON

        Parameters:
            json_data (dict): JSON for the extraction of Summaries

        Returns:
            List of summaries extracted
        """
        ret_arr = []
        keys_list = json_data[config.response_column]
        for key in keys_list:
            ret_arr.append(key[config.summary_col])
        return ret_arr

    def __len__(self):
        return len(self.texts)
