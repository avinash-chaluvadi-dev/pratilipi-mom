import logging
import os
from typing import Dict, Union

from torch.utils.data.dataloader import DataLoader

from .. import config
from ..utils import utils_tools
from .data_loader import SummarizerDataset
from .model import SummarizerBackbone
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


class SummarizerEngine:
    """
    Summarizer Engine Class - wrapper class for encapsulating the train, evaluate and serve function of the
    Summarizer package

    Attributes:
        model: Summarizer Backbone model
        dataset: TransformerModelDataset for input data
        data_loader: Data loader/ batcher for input data

    """

    def __init__(
        self,
        model: SummarizerBackbone,
        dataset: SummarizerDataset,
        data_loader: DataLoader,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader

    def train(self, func_test: bool = False) -> None:
        """
        Training method for Summarizer. Saves the model after training of model is completed

        Parameters:
            func_test: True for functional testing of summarizer package

        Returns:
            None

        """
        train_model(self.model, train_data=self.data_loader, func_test=func_test)
        return None

    def evaluate(self, x_test=None, y_test=None) -> Dict[str, Union[float, str]]:
        """
        Evaluation method for Summarizer

        Parameters:
            x_test: Input texts for evaluation
            y_test: summaries corresponding to input texts

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
                y_test = self.dataset.summaries
        return evaluate(self.model, x_test, y_test)

    def serve(self, save_result: bool = False) -> Dict:
        """
        Serving method for Summarizer

        Parameters:
            save_result: default False
                Whether to save the result locally (True for functional tests)

        Returns:
            The output JSON containing process status and details (summaries).

        """
        try:
            logging.debug("Generating predictions for input texts...")
            machine_summaries = self.model.predict(self.data_loader)
            logging.debug("Prediction Done...")

            json_data = self.dataset.json_data

            # Update the output dict for the API Plugin
            logging.debug("Generating response dictionary for API plugin")
            response = utils_tools.get_response(
                machine_summaries, json_data, status="Success"
            )

            # Check if serving is run in the save mode, and store output locally.
            if save_result is True:
                logging.info("Saving results locally...")
                utils_tools.save_prediction(self.dataset.texts, machine_summaries)

        except Exception:
            response = utils_tools.get_response(
                [], self.dataset.json_data, status="Error"
            )

            # Print the execution traceback for the exception raised
            logging.exception(
                "Exception occurred while serving the Summarizer classifier model",
                exc_info=True,
            )

        return response
