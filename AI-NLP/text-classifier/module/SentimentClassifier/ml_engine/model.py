import logging
import os

import torch
import transformers
from torch import nn

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


class SentimentBackbone(nn.Module):
    """
    T5 Fine Tuner class for Fine tuning of the T5 model
    Arguments:
        model_path: T5 model name or model path
    """

    def __init__(self, model_name: str = config.TRANSFORMER_MODEL):
        super(SentimentBackbone, self).__init__()

        if model_name == config.TRANSFORMER_MODEL:
            logging.debug("Bert - Uncased Pretrained Model to be downloaded ...")
            logging.debug("[INFO] Downloading Model..")
            self.model = transformers.BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=config.NUM_LABELS,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=True,
            )
            self.tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
            utils_tools.save_model(config.TRANSFORMER_MODEL, self.model, self.tokenizer)
            self.device = torch.device(config.DEVICE)
            self.optimizer = transformers.AdamW(self.model.parameters(), lr=5e-5)

        else:
            is_saved, path = utils_tools.is_model_saved(model_name)
            if is_saved:
                try:
                    self.model = config.model
                    self.tokenizer = config.tokenizer
                    self.device = torch.device(config.DEVICE)
                    self.optimizer = config.optimizer
                except Exception as error:
                    self.model = (
                        transformers.BertForSequenceClassification.from_pretrained(
                            model_name,
                            num_labels=config.NUM_LABELS,
                            output_attentions=False,
                            output_hidden_states=False,
                            output_scores=True,
                        )
                    )
                    self.tokenizer = transformers.BertTokenizer.from_pretrained(
                        model_name
                    )
                    utils_tools.save_model(
                        config.TRANSFORMER_MODEL, self.model, self.tokenizer
                    )
            else:
                logging.error("No Fine-Tuned to be found in the respective directory!!")
                raise ValueError(
                    "No Fine-Tuned to be found in the respective directory!!"
                )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def tokenize(self, sentence, **kwargs):
        """
        Arguments:
            Sentence
            Arguments required by T5 Tokenizer
        Returns:
            Tokenized Sentence
        """
        if isinstance(sentence, str):
            return self.tokenizer.batch_encode_plus([sentence], **kwargs)
        else:
            return [
                self.tokenizer.batch_encode_plus([sent], **kwargs) for sent in sentence
            ]


class SentimentClassifier2(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier2, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.TRANSFORMER_MODEL)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

    def tokenize(self, sentence, **kwargs):
        """
        Arguments:
            Sentence
            Arguments required by T5 Tokenizer
        Returns:
            Tokenized Sentence
        """
        if isinstance(sentence, str):
            return self.tokenizer.batch_encode_plus([sentence], **kwargs)
        else:
            return [
                self.tokenizer.batch_encode_plus([sent], **kwargs) for sent in sentence
            ]
