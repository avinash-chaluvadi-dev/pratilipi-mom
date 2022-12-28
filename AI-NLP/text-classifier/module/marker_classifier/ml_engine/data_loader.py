import torch
from torch.utils.data import Dataset

from .. import config
from ..utils import custom_logging


class MarkerDataset(Dataset):
    """

    MarkerDataset class - to load the dataset used the __getitem__ fashion supported by the Pytorch.
    The loader supports the JSON and the csv format for parsing the input to the network.
        :param mode: mode of Marker classifier model (train, eval or serve)
        :param text: input text for train, eval and serve components
        :param marker: output marker for train and eval components
        :param func_test: True for functional testing of package
        :param tokenizer: tokenizer from huggingface library

    """

    logger = custom_logging.get_logger()

    def __init__(self, text, func_test, tokenizer, marker=None, mode="serve"):
        self.mode = mode
        self.text = text
        self.marker = marker
        self.func_test = func_test
        self.tokenizer = tokenizer

    @classmethod
    def from_dataframe(cls, dataframe, mode, tokenizer, func_test=None):
        """

        Converts dataframe to numpy array of text and marker.
            :param dataframe: Input dataframe for the Marker Classifier
            :param mode: mode of Marker classifier model (train, eval or serve)
            :param tokenizer: tokenizer from huggingface library
            :param func_test: True for functional testing of package
            :return numpy.ndarray: array of text and marker if mode is train or eval.

        """
        if mode in ["train_eval", "train", "eval"] or (
            mode == "package_test" and func_test in ["train", "eval"]
        ):
            text = dataframe.loc[:, config.params.get("csv").get("text_column")].values
            marker = dataframe.loc[
                :, config.params.get("csv").get("marker_column")
            ].values
            return cls(
                text=text,
                func_test=func_test,
                tokenizer=tokenizer,
                marker=marker,
                mode=mode,
            )

        elif mode == "serve" or (mode == "package_test" and func_test == "serve"):
            text = dataframe.loc[:, config.params.get("data").get("text")].values
            return cls(text=text, func_test=func_test, tokenizer=tokenizer, mode=mode)
        else:
            cls.logger.exception("Invalid mode for dataset creation..")

    @classmethod
    def from_json(cls, json_data, tokenizer, mode="serve", func_test=None):
        """

        Converts Json to numpy array of text and marker.
            :param json_data: Input json for the Marker Classifier
            :param mode: mode of Marker classifier model (train, eval or serve)
            :param tokenizer: tokenizer from huggingface library
            :param func_test: True for functional testing of package
            :return numpy.ndarray: array of text and marker if mode is train or eval.

        """
        if mode == "serve" or (mode == "package_test" and func_test == "serve"):
            text = []
            for index in range(len(json_data)):
                for segment in json_data[index].get(
                    config.params.get("json").get("response_key")
                ):
                    text.append(segment.get(config.params.get("json").get("text_key")))
            return cls(text=text, func_test=func_test, tokenizer=tokenizer, mode=mode)

        else:
            cls.logger.exception("Invalid mode for dataset creation..")

    def __getitem__(self, item):
        """

        Returns tokenized tensors for text and marker(if mode is train or eval) for the given index.
            :param: item: index to fetch the data.
            :returns dict: dictionary of tensors containing input_ids, attention_mask,
                           token_type_ids and marker(output if mode is train or eval)

        """
        if self.mode in ["train_eval", "train", "eval"] or (
            self.mode == "package_test" and self.func_test in ["train", "eval"]
        ):
            text = str(self.text[item])
            processed_text = " ".join(text.split())
            marker = self.marker[item]
            inputs = self.tokenizer.encode_plus(
                processed_text,
                None,
                add_special_tokens=True,
                max_length=config.MAX_LEN,
                truncation=True,
            )
            padding_length = config.MAX_LEN - len(inputs.get("input_ids"))
            input_ids = inputs.get("input_ids") + ([0] * padding_length)
            attention_mask = inputs.get("attention_mask") + ([0] * padding_length)
            token_type_ids = inputs.get("token_type_ids") + ([0] * padding_length)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "marker": torch.tensor(
                    config.CLASSIFICATION_LABELS.index(marker.title()),
                    dtype=torch.long,
                ),
            }
        elif self.mode == "serve" or (
            self.mode == "package_test" and self.func_test == "serve"
        ):
            text = str(self.text[item])
            text = " ".join(text.split())
            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=config.MAX_LEN,
                truncation=True,
            )
            padding_length = config.MAX_LEN - len(inputs.get("input_ids"))
            input_ids = inputs.get("input_ids") + ([0] * padding_length)
            attention_mask = inputs.get("attention_mask") + ([0] * padding_length)
            token_type_ids = inputs.get("token_type_ids") + ([0] * padding_length)
            return {
                "text": text,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            }

    def __len__(self):
        """

        Returns length of dataset
            :returns int: length of dataset

        """
        return len(self.text)
