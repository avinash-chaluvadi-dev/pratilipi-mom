import logging
import os
from typing import List, Union

import numpy as np
import torch
import transformers
from torch import nn
from torch.utils.data.dataloader import DataLoader

from .. import config
from ..utils import utils_tools
from .data_loader import HeadlinerDataset

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


class HeadlinerBackbone(nn.Module):
    """
    HeadlinerBackbone - Backbone model class for Text Headliner

    Attributes:
        model_name: model name for loading the text headliner model (fine tuned or pretrained)

    """

    def __init__(self, model_name: str = config.HEADLINER_MODEL):
        super(HeadlinerBackbone, self).__init__()

        if model_name is config.HEADLINER_MODEL:
            model_name = config.BASE_MODEL
        # Checking if model is already saved
        is_saved, path = utils_tools.is_model_saved(model_name)

        if is_saved:
            try:  # Try loading the model
                self.model = config.model
                self.tokenizer = config.tokenizer

            except Exception as error:  # Downloading the model in case we get error loading the saved model
                logging.warning(f"[ERROR] {error}. Downloading the Model now...")
                self.model = transformers.T5ForConditionalGeneration.from_pretrained(
                    model_name
                )
                self.tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
                utils_tools.save_model(model_name, self.model, self.tokenizer)
        else:
            model_name = config.HEADLINER_MODEL
            logging.info(f"[{model_name}] Downloading Model..")  # Downloading the model
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(
                model_name
            )
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
            utils_tools.save_model(config.BASE_MODEL, self.model, self.tokenizer)
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and config.DEVICE == "cuda"
            else torch.device("cpu")
        )
        self.optimizer = config.optimizer
        self.model_name = model_name

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def tokenize(self, sentence: str, **kwargs) -> List[str]:
        """
        Tokenizing given sentence/s using the T5 tokenizer

        Parameters:
            sentence (str, list): string or list of strings
            kwargs: other keyword arguments (kwargs) that are passed to tokenizer

        Returns:
            tokenized string which is the output from tokenizer

        """
        if isinstance(sentence, str):
            return self.tokenizer.batch_encode_plus([sentence], **kwargs)
        else:
            return [
                self.tokenizer.batch_encode_plus([sent], **kwargs) for sent in sentence
            ]

    def _get_prediction(self, data: DataLoader):
        """
        Arguments:
            data: data loader/ batcher for the test Dataset
        Returns:
            headline of the input dataset passed
        """
        device = self.device
        self.model.to(device)
        self.model.eval()
        labels_pred = []
        ind = 0

        for batch in data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                headline_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=3,
                    no_repeat_ngram_size=2,
                    repetition_penalty=2.5,
                    max_length=config.MAX_LENGTH,
                    early_stopping=False,
                )
            output = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in headline_ids
            ]
            if config.BATCH_SIZE != 1:
                logging.info(
                    f"[{self.model_name}] Sample {ind + 1} - {ind + config.BATCH_SIZE} Done."
                )
            else:
                logging.info(f"[{self.model_name}] Sample {ind + 1} Done.")
            ind += config.BATCH_SIZE
            labels_pred.extend(output)

            # Clearing batch data from memory
            del batch

            # Clearing headliner output from memory
            del headline_ids

        return [". ".join(labels_pred)]

    def predict(self, data: Union[DataLoader, List]) -> List:
        """
        Wrapper method for getting the prediction for input texts

        Parameters:
            data: Input data for getting the prediction

        Returns:
            list of headlines corresponding to the input texts

        """
        if isinstance(data, (list, np.ndarray)):
            # Creating Dataloader if input data is in list format
            dataset = HeadlinerDataset(self, is_train=False)
            data = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        return self._get_prediction(data)
