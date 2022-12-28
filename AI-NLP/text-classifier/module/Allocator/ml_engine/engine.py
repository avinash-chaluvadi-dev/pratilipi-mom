import logging
import os
from typing import Dict, List

import numpy as np

from .. import config
from ..utils import utils_tools
from .train import evaluate, get_features, kfold_cross_validation, train_model

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


class AllocatorEngine:
    """
    Allocator Engine Class - wrapper class for encapsulating the train, evaluate and serve function of the
    Allocator Classifier

    Attributes:
        model: Transformer model for Feature extraction or Sequence Classification
        data_loader: Data loader/ batcher for input data
        dataset: TransformerModelDataset for input data
        sequence_classification: Whether to use transformer model for sequence classification
                                (False value will mean using transformer model for feature extraction)

    """

    def __init__(
        self, model, data_loader, dataset, sequence_classification=False
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.dataset = dataset
        self.sequence_classification = sequence_classification

    def decode_labels(self, labels):
        """
        Decoding labels

        Parameters:
            labels: list of predictions values

        Returns: list of values decoded

        """
        if isinstance(labels, (float, int)):
            labels = [labels]
        return [config.INVERSE_LABEL_DICT[label] for label in labels]

    def train(self, kfold_training: bool = True, func_test: bool = False):
        """
        Training method for Allocator (Deadline and Escalation classifier). Saves the model after training of model is completed

        Parameters:
            kfold_training: Whether to do the kfold training or not
            func_test: True for functional testing of Allocator (Deadline and Escalation classifier) package

        """
        # Training using K fold validation by training on k-1 fold and evaluating on 1 fold
        if kfold_training is True:
            # Checking if data samples are less than number of splits in K Fold
            if len(self.dataset) < config.N_SPLITS:
                logging.exception(
                    "Number of samples for training is less than number of splits in K Fold."
                    "Set KFOLD_TRAINING=False for simple training..."
                )
                raise ValueError(
                    "Number of samples for training is less than N_SPLITS in K Fold"
                )

            logging.info("Training with K fold cross validation started...")
            return kfold_cross_validation(
                self.model,
                self.dataset,
                is_train=True,
                sequence_classification=self.sequence_classification,
                func_test=func_test,
            )
        # Training on the complete input data
        else:
            return train_model(
                self.model,
                self.data_loader,
                sequence_classification=self.sequence_classification,
                func_test=func_test,
            )

    def evaluate(
        self,
        x_test=None,
        y_test=None,
        kfold_validation: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluation method for Allocator (Deadline and Escalation classifier)

        Parameters:
            x_test: Input texts for evaluation
            y_test: Labels corresponding to input texts
            kfold_validation: Whether or not to use k fold cross validation

        Returns:
            Dictionary of Evaluation Scores

        """
        scores = {}

        # Model Evaluation using K Fold evaluation (Training on K-1 Fold and evaluating on 1 Fold)
        if kfold_validation is True:
            # Checking if num samples is less than Num splits in K Fold
            if len(self.dataset) < config.N_SPLITS:
                logging.exception(
                    "Number of samples for evaluation is less than number of splits in K Fold. "
                    "Set KFOLD_VALIDATION=False for simple evaluation"
                )
                raise ValueError(
                    "Number of samples for evaluation is less than N_SPLITS in K Fold"
                )
            scores = kfold_cross_validation(
                transformer_model=self.model,
                data=self.dataset,
                is_train=False,
                sequence_classification=self.sequence_classification,
            )
        else:
            if (
                x_test is None or y_test is None
            ):  # Evaluating on local data, if evaluation data is not passed
                if self.dataset is None:
                    scores = {
                        "Scores": "No Evaluation data found. Please pass testing data for evaluation scores"
                    }
                    logging.warning(
                        "No Evaluation data found for evaluation process..."
                    )
                else:
                    x_test = self.dataset.texts
                    y_test = self.dataset.labels
                    scores = evaluate(self.model, x_test=x_test, y_test=y_test)
            else:  # if evaluation data is present
                scores = evaluate(self.model, x_test=x_test, y_test=y_test)
        return scores

    def serve(self, save_result: bool = False) -> Dict[str, List]:
        """
        Serving method for Allocator (Deadline and Escalation classifier)

        Parameters:
            save_result: default False
                Whether or not to save the result locally (True for functional tests)

        Returns:
            The output JSON containing process status and details.

        """
        try:
            logging.debug("Generating predictions for input texts...")
            if self.sequence_classification is True:
                labels_prob = self.model.predict(self.data_loader)
            else:
                # loading ML model for prediction
                ml_model = utils_tools.pickle_load(config.BASE_ML_MODEL)

                # getting features from input data
                features, _ = get_features(self.model, self.dataset, is_train=False)
                labels_prob = ml_model.predict(features)
            logging.debug("Decoding labels of prediction output...")
            labels_pred = [np.argmax(array) for array in labels_prob]

            # Storing chunk IDs
            chunk_ids = self.dataset.chunk_ids
            # Getting confidence scores for each class
            confidence_scores = [
                round(array[np.argmax(array)] * 100, 2) for array in labels_prob
            ]
            labels = self.decode_labels(labels_pred)
            logging.debug("Prediction Done...")

            # If preprocessing is done, then getting the original texts
            text_list = (
                self.dataset.original_texts
                if config.PREPROCESS_TEXT is True
                else self.dataset.texts
            )
            json_data = self.dataset.json_data

            # Update the output dict for the API Plugin
            logging.debug("Generating response dictionary for API plugin")
            response = utils_tools.get_response(
                labels, confidence_scores, json_data, status="Success"
            )

            # Check if serving is run in the save mode, and store output locally.
            if save_result is True:
                logging.debug("Saving results locally...")
                utils_tools.save_prediction(text_list, labels)

        except Exception:
            response = utils_tools.get_response(
                [], [], self.dataset.json_data, status="Error"
            )

            # Print the execution traceback for the exception raised
            logging.exception(
                "Exception occurred while serving the Allocator (Deadline and Escalation classifier) model",
                exc_info=True,
            )

        return response
