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
from .data_loader import SummarizerDataset

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


class SummarizerBackbone(nn.Module):
    """
    SummarizerBackbone - Backbone model class for Text Summarizer

    Attributes:
        model_name: model name for loading the text summarizer model (fine tuned or pretrained)

    """

    def __init__(self, model_name: str = config.SUMMARIZATION_MODEL) -> None:
        super(SummarizerBackbone, self).__init__()

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
            logging.info(f"[{model_name}] Downloading Model..")  # Downloading the model
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(
                model_name
            )
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
            utils_tools.save_model(model_name, self.model, self.tokenizer)
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
        Tokenizing given sentence/ s using the T5 tokenizer

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

    def _get_prediction(self, data):
        """
        Arguments:
            data: data loader/ batcher for the test Dataset
        Returns:
            summaries of the input dataset passed
        """
        logging.debug("Prediction started...")
        self.model.to(self.device)  # Loading the model into GPU/ CPU
        self.model.eval()
        summary_pred = []
        ind = 0
        for batch in data:
            input_ids = batch["input_ids"].to(
                self.device
            )  # Loading the data into GPU/ CPU
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                # Generating ids for the input dataset
                summary_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=config.NUM_BEAMS,
                    no_repeat_ngram_size=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    max_length=config.OUTPUT_LENGTH,
                    early_stopping=True,
                )

            # Decoding the ids using the tokenizer
            output = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in summary_ids
            ]
            if config.BATCH_SIZE != 1:
                logging.info(
                    f"[SUMMARIZER] Sample {ind + 1} - {ind + config.BATCH_SIZE} Done."
                )
            else:
                logging.info(f"[SUMMARIZER] Sample {ind + 1} Done.")
            ind += config.BATCH_SIZE
            summary_pred.extend(output)

            # Clearing batch data from memory
            del batch

            # Clearing sumarizer output from memory
            del summary_ids

        return summary_pred

    def predict(self, data: Union[DataLoader, List]) -> List[str]:
        """
        Wrapper method for getting the prediction for input texts

        Parameters:
            data: Input data for getting the prediction

        Returns:
            list of summaries corresponding to the input texts

        """
        if isinstance(data, (list, np.ndarray)):
            # Creating Dataloader if input data is in list format
            dataset = SummarizerDataset(self, text_list=data, is_train=False)
            data = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        return self._get_prediction(data)


class PegaususParaphraser(nn.Module):
    """
    Pegasus Model for Paraphrasing Sentences
    Arguments:
        model_path: Pegasus Paraphraser model_name or path
    """

    def __init__(self, model_path: str = config.PARAPHRASING_MODEL) -> None:
        super(PegaususParaphraser, self).__init__()
        self.paraphraser = transformers.PegasusForConditionalGeneration.from_pretrained(
            model_path
        )
        self.tokenizer = transformers.PegasusTokenizer.from_pretrained(model_path)
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and config.DEVICE == "cuda"
            else torch.device("cpu")
        )
        if not utils_tools.is_model_saved(model_path):
            utils_tools.save_model(model_path, self.model, self.tokenizer)

    def forward(self, text: str):
        batch = self.tokenizer.prepare_seq2seq_batch(
            [text], truncation=True, padding="longest", max_length=128
        ).to(self.device)
        translated = self.paraphraser.generate(
            **batch,
            max_length=config.INPUT_LENGTH,
            num_beams=config.NUM_BEAMS,
            temperature=config.TEMPERATURE,
        )
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text[0]
