import logging
import os
from typing import Dict

from torch.utils.data.dataloader import DataLoader

from .. import config
from ..utils import utils_tools
from .data_loader import HeadlinerDataset
from .model import HeadlinerBackbone
from .train import evaluate, train_model

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


class HeadlinerEngine:
    """
    Headliner Engine Class - wrapper class for encapsulating the train, evaluate and serve function of the
    Headliner package

    Attributes:
        model: Headliner Backbone model
        dataset: TransformerModelDataset for input data
        data_loader: Data loader/ batcher for input data

    """

    def __init__(
        self,
        model: HeadlinerBackbone,
        dataset: HeadlinerDataset,
        data_loader: DataLoader,
    ):
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader

    def train(self, func_test: bool = False):
        """
        Training method for Headliner. Saves the model after training of model is completed

        Parameters:
            func_test: True for functional testing of Headliner package

        """
        train_model(self.model, train_data=self.data_loader, func_test=func_test)

    def evaluate(self, x_test=None, y_test=None) -> Dict[str, float]:
        """
        Evaluation method for Headliner

        Parameters:
            x_test: Input texts for evaluation
            y_test: headlines corresponding to input texts

        Returns:
            Dictionary of Evaluation Scores

        """

        if x_test is None:
            if not self.dataset or len(self.dataset) == 0:
                scores = {
                    "Scores": "No Evaluation data found. Please pass testing data for evaluation scores"
                }
                logging.warning("No Evaluation data found for evaluation process...")
                return scores
            else:
                x_test = self.dataset.texts
                y_test = self.dataset.headline
        return evaluate(self.model, x_test, y_test)

    def serve(self, save_result: bool = False):
        """
        Serving method for Headliner

        Parameters:
            save_result: default False
                Whether or not to save the result locally (True for functional tests)

        Returns:
            The output JSON containing process status and details (headlines).

        """
        try:
            logging.debug("Generating predictions for input texts...")
            machine_headlines = self.model.predict(self.data_loader)[0]
            combined_text = ". ".join(self.dataset.texts)
            json_data = self.dataset.json_data

            logging.debug("Prediction Done...")

            # Update the output dict for the API Plugin
            logging.debug("Generating response dictionary for API plugin")
            response = utils_tools.get_response(
                machine_headlines, json_data, status="Success"
            )

            # Check if serving is run in the save mode, and store output locally.
            if save_result is True:
                logging.info("Saving results locally...")
                utils_tools.save_prediction([combined_text], [machine_headlines])

        except Exception:
            response = utils_tools.get_response(
                "", self.dataset.json_data, status="Error"
            )

            # Print the execution traceback for the exception raised
            logging.exception(
                "Exception occurred while serving the Headliner classifier model",
                exc_info=True,
            )
        return response
