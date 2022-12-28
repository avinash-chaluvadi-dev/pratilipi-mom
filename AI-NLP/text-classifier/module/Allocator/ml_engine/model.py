import logging
import os
from typing import Dict, List, Union

import numpy as np
import torch
import transformers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.svm import SVC
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)
from xgboost import XGBClassifier

from .. import config
from ..utils import utils_tools
from .data_loader import TransformerModelDataset

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


class BaseModel:
    """
    BaseModel Class which contains common method and attributes that can be inherited by other model classes

    """

    def __init__(self):
        super().__init__()

    def _train(self, train_data, test_data=None):
        """
        Training of the model

        Parameters:
            train_data: tuple consisting of train data and train labels
            test_data: tuple consisting of test data and test labels

        Returns:
            evaluation scores from evaluate() function

        Returns:


        """
        logging.debug(f"Training of {self.model_name} started...")
        x_train, y_train = train_data
        self.model.fit(x_train, y_train)
        logging.debug(f"Training of {self.model_name} completed...")

        # Model evaluation to be done if evaluation data is present
        if test_data is not None:
            x_test, y_test = test_data
            return self.evaluate(x_test, y_test)
        else:
            return {"Scores": "No Evaluation Data found!"}

    def evaluate(
        self,
        x_test: Union[DataLoader, List],
        y_test: List,
    ) -> Dict[str, float]:
        """
        Evaluating the model

        Parameters:
            x_test: evaluation data
            y_test: evaluation labels corresponding to X_test

        Returns:
            Scores: dictionary of different evaluation scores (f1, accuracy, precision, recall, roc_auc)

        """
        y_pred_probs = self.predict(
            x_test
        )  # probabilities for each class for the given texts
        y_pred = [np.argmax(array) for array in y_pred_probs]

        # Getting confidence scores for each class
        confidence_scores = [
            round(array[np.argmax(array)] * 100, 2) for array in y_pred_probs
        ]
        scores = {
            "f1_score": f1_score(y_pred=y_pred, y_true=y_test, average="macro"),
            "accuracy_score": accuracy_score(y_pred=y_pred, y_true=y_test),
            "precision_score": precision_score(
                y_pred=y_pred, y_true=y_test, average="macro"
            ),
            "recall_score": recall_score(y_pred=y_pred, y_true=y_test, average="macro"),
        }
        try:
            # ROC AUC score gives error if only one class label is present in output
            roc_auc = roc_auc_score(y_test, y_pred, average="macro", multi_class="ovr")
            scores["roc_auc_score"] = roc_auc
        except Exception:
            logging.exception("ROC_AUC_Score Error")

        logging.info(f"[{self.model_name}] Scores:\n{scores}")
        return scores

    def _load_transformer_model(self, model_name, sequence_classification=False):
        """
        Loading the transformer model

        Parameters:
            model_name(str): transformer model name for loading the transformer model
            sequence_classification (bool): whether or not to load the Sequence Classification model

        Returns:
            model: Transformer model
            tokenizer: Transformer tokenizer

        """

        # AutoClasses helps in automatically retrieving the relevant model
        # given the name/path to the pretrained weights/config/vocabulary.

        if not sequence_classification:
            transformer_model = AutoModel
        else:
            transformer_model = AutoModelForSequenceClassification

        is_saved, path = utils_tools.is_model_saved(
            model_name
        )  # Checking if model is saved previously or not
        if is_saved:
            try:  # Try loading the model
                logging.info(f"[{model_name}] Loading Model..")
                model_path = path
                model = transformer_model.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as error:  # Downloading the model in case we get error loading the saved model
                logging.warning(f"[ERROR] {error}")
                logging.info(f"[{model_name}] Downloading Model..")
                model = transformer_model.from_pretrained(
                    model_name, num_labels=config.NUM_LABELS
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                utils_tools.save_model(model_name, model, tokenizer)
        else:
            logging.info(f"[{model_name}] Downloading Model..")  # Downloading the model
            model = transformer_model.from_pretrained(
                model_name, num_labels=config.NUM_LABELS
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            utils_tools.save_model(model_name, model, tokenizer)
        return model, tokenizer

    def _tokenize(self, sentence, **kwargs):
        """
        Tokenizing given sentence/ s using the transformer tokenizer

        Parameters:
            sentence (str, list): string or list of strings
            kwargs: other keyword arguments(kwargs) that are passed to tokenizer

        Returns:
            tokenized string which is the output from tokenizer

        """
        if isinstance(sentence, str):
            return self.tokenizer.batch_encode_plus([sentence], **kwargs)
        else:
            return [
                self.tokenizer.batch_encode_plus([sent], **kwargs) for sent in sentence
            ]


class SVMClassifier(BaseModel):
    """
    SVMClassifier - wrapper class for SVC (sklearn.svm.SVC)
    It inherits BaseModel Class

    """

    def __init__(self):
        super(SVMClassifier, self).__init__()
        self.model = SVC()
        self.model_name = "SVM Classifier"

    def train(self, train_data: tuple, test_data: tuple) -> dict:
        return self._train(train_data, test_data)

    def predict(self, x_test):
        predictions = self.model.predict_proba(x_test)
        return [utils_tools.softmax(pred) for pred in predictions]


class RandomForestClassifierModel(BaseModel):
    """
    RandomForestClassifierModel - wrapper class for RandomForestClassifier (sklearn.ensemble.RandomForestClassifier)
    It inherits BaseModel Class

    """

    def __init__(self):
        super(RandomForestClassifierModel, self).__init__()
        self.model = RandomForestClassifier(n_estimators=70)
        self.model_name = "Random Forest Classifier"

    def train(self, train_data: tuple, test_data: tuple) -> dict:
        return self._train(train_data, test_data)

    def predict(self, x_test):
        predictions = self.model.predict_proba(x_test)
        return [utils_tools.softmax(pred) for pred in predictions]


class XGBoostClassifier(BaseModel):
    """
    XGBoostClassifier - wrapper class for XGBClassifier (xgboost.XGBClassifier)
    It inherits BaseModel Class

    """

    def __init__(self):
        super(XGBoostClassifier, self).__init__()
        self.model = XGBClassifier()
        self.model_name = "XGBoost Classifier"

    def train(self, train_data: tuple, test_data: tuple) -> dict:
        return self._train(train_data, test_data)

    def predict(self, x_test):
        predictions = self.model.predict_proba(x_test)
        return [utils_tools.softmax(pred) for pred in predictions]


class SGDClassifierModel(BaseModel):
    """
    SGDClassifierModel - wrapper class for SGDClassifier (sklearn.linear_model.SGDClassifier)
    It inherits BaseModel Class

    """

    def __init__(self):
        super(SGDClassifierModel, self).__init__()
        self.model = SGDClassifier(loss="log")
        self.model_name = "SGD Classifier"

    def train(self, train_data: tuple, test_data: tuple) -> dict:
        return self._train(train_data, test_data)

    def predict(self, x_test):
        predictions = self.model.predict_proba(x_test)
        return [utils_tools.softmax(pred) for pred in predictions]


class MLBackbone:
    def __init__(self, model_name: str):
        self.model_list = [
            SGDClassifierModel,
            RandomForestClassifierModel,
            SVMClassifier,
            XGBoostClassifier,
        ]
        self.model = self.find_model(model_name)
        self.model_name = model_name

    def find_model(self, model_name):
        clf_model = None
        for model in self.model_list:
            model = model()
            if model.model_name == model_name:
                clf_model = model
                break
            del model
        if clf_model is None:
            logging.exception(f"Model Name {model_name} is not from {self.model_list}")
            raise ValueError(f"Please provide model name from {self.model_list}")
        return clf_model

    def train(self, train_data: tuple, test_data: tuple) -> dict:
        return self.model.train(train_data, test_data)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(
        self,
        x_test: Union[DataLoader, List],
        y_test: List,
    ) -> Dict[str, float]:
        return self.model.evaluate(x_test, y_test)


class TransformerForFeatureExtraction(BaseModel, nn.Module):
    """
    TransformerForFeatureExtraction class for extracting features from given tokenized texts
    It inherits BaseModel Class and torch.nn.Module class

    Attributes:
        model_name: Transformer model name from config.TRANSFORMER_MODEL_LIST
    """

    def __init__(self, model_name: str) -> None:
        super(TransformerForFeatureExtraction, self).__init__()

        # loading the model and tokenizer
        self.model, self.tokenizer = self._load_transformer_model(model_name)
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and config.DEVICE == "cuda"
            else torch.device("cpu")
        )
        self.model_name = model_name

    def forward(self, **kwargs):
        with torch.no_grad():
            output = self.model(**kwargs)
        last_hidden_state = output[0]
        # Return [CLS] token from the last hidden state
        return last_hidden_state[:, 0, :].cpu().numpy()

    def tokenize(self, sentence, **kwargs):
        return self._tokenize(sentence, **kwargs)


class TransformerForSequenceClassification(BaseModel, nn.Module):
    """
    TransformerForSequenceClassification Class for Text (sequence) classification task

    Attributes:
        model_name: Transformer model name from config.TRANSFORMER_MODEL_LIST or saved transformer model name

    """

    def __init__(self, model_name: str):
        super(TransformerForSequenceClassification, self).__init__()
        self.model, self.tokenizer = self._load_transformer_model(
            model_name, sequence_classification=True
        )
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and config.DEVICE == "cuda"
            else torch.device("cpu")
        )
        self.optimizer = transformers.AdamW(
            self.model.parameters(), lr=config.LEARNING_RATE
        )
        self.model_name = model_name

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def tokenize(self, sentence: Union[str, List], **kwargs):
        return self._tokenize(sentence, **kwargs)

    def predict(self, data: Union[DataLoader, List]) -> List[np.ndarray]:
        """
        Prediction of labels for given list of texts

        Parameters:
            data: evaluation / serving data

        Returns:
            list of probabilities of each class for given list of texts

        """
        if isinstance(data, (list, np.ndarray)):
            # Creating Dataloader if input data is in list format
            dataset = TransformerModelDataset(self, text_list=data, is_train=False)
            data = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        device = self.model.device
        self.model.to(device)
        self.model.eval()
        y_pred = []
        try:
            for index, encodings in enumerate(data):
                # loading data into GPU/ CPU
                batch_data = {key: data.to(device) for key, data in encodings.items()}
                outputs = self.model(**batch_data)
                logits = outputs.logits  # Output logits
                for logit in logits:
                    probs = logit.softmax(0)  # getting probabilities for each class
                    probs = np.array([float(prob) for prob in probs])
                    y_pred.append(probs)
            return y_pred
        except Exception:
            logging.exception(
                "Error in predicting the texts from Allocator (Deadline and Escalation classifier) model."
            )
            raise


class AllocatorBackbone(nn.Module):
    """
    AllocatorBackbone - Backbone model class for Allocator (Deadline and Escalation classifier)

    Attributes:
        model_name: Transformer model name from config.TRANSFORMER_MODEL_LIST or saved transformer model name
        sequence_classification: Whether to use transformer model for sequence classification
                                (False value will mean using transformer model for feature extraction)

    """

    def __init__(self, model_name: str, sequence_classification: bool = False) -> None:
        super(AllocatorBackbone, self).__init__()
        if sequence_classification is False:
            model_object = TransformerForFeatureExtraction(model_name=model_name)
        else:
            model_object = TransformerForSequenceClassification(model_name=model_name)
            self.optimizer = model_object.optimizer
            self.model_name = model_object.model_name
        self.model = model_object.model
        self.tokenizer = model_object.tokenizer
        self.device = model_object.device
        self.model_object = model_object
        self.sequence_classification = sequence_classification

    # Wrapper function for Transformer model forward method
    def forward(self, **kwargs):
        return self.model_object.forward(**kwargs)

    # Wrapper function for Base model tokenize method
    def tokenize(self, sentence: Union[str, List], **kwargs) -> List:
        return self.model_object.tokenize(sentence, **kwargs)

    # Wrapper function for Transformer/ ML model predict method
    def predict(self, eval_data: Union[DataLoader, List]) -> List:
        return self.model_object.predict(data=eval_data)

    # Wrapper function for Base model evaluate method
    def evaluate(
        self,
        x_test: Union[DataLoader, List],
        y_test: List,
    ) -> Dict[str, float]:
        return self.model_object.evaluate(x_test=x_test, y_test=y_test)
