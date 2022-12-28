import torch
from torch.utils.data import Dataset

from .. import config
from ..utils import custom_logging


class LabelDataset(Dataset):
    """
    LabDataset class - to load the dataset used the __getitem__ fashion supported by the Pytorch.
    The loader supports the JSON and the csv format for parsing the input to the network.
    """

    logger = custom_logging.get_logger()

    def __init__(self, text, func_test, tokenizer, label=None, mode="serve"):
        self.mode = mode
        self.text = text
        self.label = label
        self.func_test = func_test
        self.tokenizer = tokenizer

    @classmethod
    def from_dataframe(cls, dataframe, mode, tokenizer, func_test=None):
        """
        Converts dataframe to numpy array of text and lables.

        """
        if mode in ["train_eval", "train", "eval"] or (
            mode == "package_test" and func_test in ["train", "eval"]
        ):
            text = dataframe.loc[:, config.params.get("csv").get("text_column")].values
            label = dataframe.loc[
                :, config.params.get("csv").get("label_column")
            ].values
            return cls(
                text=text,
                func_test=func_test,
                tokenizer=tokenizer,
                label=label,
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
        Converts dataframe to numpy array of text and labels.

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

        Returns tokenized tensors for text and label(if mode is train or eval) for the given index.
            :param: item: index to fetch the data.
            :returns dict: dictionary of tensors containing input_ids, attention_mask,
                           token_type_ids and labels(output if mode is train or eval)

        """
        if self.mode in ["train_eval", "train", "eval"] or (
            self.mode == "package_test" and self.func_test in ["train", "eval"]
        ):
            text = str(self.text[item])
            text = " ".join(text.split())
            label = self.label[item]
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
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "label": torch.tensor(
                    config.CLASSIFICATION_LABELS.index(label.strip()),
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
        """
        return len(self.text)
