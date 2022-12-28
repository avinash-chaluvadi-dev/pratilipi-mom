import json
import logging
import traceback

import numpy as np
import torch
from scipy.special import softmax

from .. import config
from ..utils import utils_tools


class SentimentEngine:
    """

    Sentiment classifier engine class: This class to encapsulate the train, serving and evaluating function of the
    Sentiment classifier
    """

    def __init__(self, model=None, data_loader=None):
        """

        :param model: The backbone model class object instantiate with weights loaded.
        """

        self.model = model
        self.data_loader = data_loader

    def train(self, save_result=False):
        """
        Sentiment Classifier training Method. This methods instantiates the data loader in the training mode and store
        the model after total training runs have been completed.
        """

        try:
            # Fetching the device object for the model training.
            device = self.model.device
            # To move the model parameters to the available device.
            self.model.to(device)

            # Call the train method from the nn.Module base class
            self.model.train()

            # Training loop start
            logging.debug("Starting the training Loop ..")
            for epoch in range(config.EPOCHS):
                logging.debug(f"[INFO] Epoch {epoch + 1} Started..")
                for index, batch in enumerate(self.data_loader):
                    logging.debug(
                        f"[INFO] Epoch {epoch + 1} Iteration {index + 1} Running.."
                    )
                    self.model.optimizer.zero_grad()
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs[0]
                    logging.debug(f"Training loss : {loss}")
                    loss.backward()
                    self.model.optimizer.step()
                logging.debug(f"[INFO] Epoch {epoch + 1} Ended..")

            # Saving the model after the training run.
            if save_result:
                utils_tools.save_model(
                    config.SAVE_FINETUNED_CLASSIFIER,
                    self.model.model,
                    self.model.tokenizer,
                )

        except Exception:
            # Print the execution traceback for the exception raised
            logging.debug("Training Exception Occured", exc_info=True)

    def evaluate(self):
        """
        :param model:
        :return:
        """
        pass

    def serve(self, save_result=False):
        """

        :return: The output JSON expected to be integrated with the model API.
        """

        try:
            if save_result:
                # Generate the Sentiment Prediction.
                logging.debug("Generating Sentiment Prediction..")
                input_text, labels_pred, conf_pred = get_sentiments(
                    self.model, self.data_loader
                )

                # Postprocessing the Sentiment Prediction.
                logging.debug("Postprocessing Sentiment Predictions..")
                pred_processed = decode_sentiments(labels_pred)

                output_dict = {
                    "transcript": input_text,
                    "classifier_output": pred_processed,
                    "confidence_score": conf_pred,
                }

                # Save results
                save_prediction(output_dict)

                # This returned results is not consumed by any code block, writing this.
                # To make it consistent with PEP-8 return statement Convention.
                return output_dict

            else:
                # Generate the Sentiment Prediction.
                logging.debug("Generating Sentiment Prediction..")
                input_text, labels_pred, conf_pred = get_sentiments(
                    self.model, self.data_loader
                )

                # Postprocessing the Sentiment Prediction.
                logging.debug("Postprocessing Sentiment Predictions..")
                pred_processed = decode_sentiments(labels_pred)

                output_dict = {
                    "transcript": input_text,
                    "classifier_output": pred_processed,
                    "confidence_score": conf_pred,
                }

                logging.debug("Generating the response dictionary for the API plugin..")
                # Update the output dict for the API Plugin
                response = {
                    "status": "Success",
                    "model": "sentiment_classifier",
                    "response": output_dict,
                }
                return response

        except Exception:

            # Add the traceback to the Details key when the serving process fails.
            logging.exception(
                "Exception encountered while serving the Sentiment Engine",
                exc_info=True,
            )
            response = {
                "status": "Error",
                "model": "sentiment_classifier",
                "response": f"{traceback.format_exc()}",
            }
            return response


# Sentiment Engine Specific Utilities
def decode_sentiments(predictions):
    """

    :param predictions: Get the python [] of raw prediction tensor from the model.
    :return: Convert the prediction probabilities into labels python [].
    """
    for ind, val in enumerate(predictions):
        predictions[ind] = config.DECODE_LABELS[val]
    return predictions


def get_sentiments(model, dataloader):
    """
    Arguments:
        model: SentimentClassifier model object
        dataloader: Dataloader object for the test Dataset
    Returns:
        summary of the test dataset
    """
    device = model.device
    model.to(device)
    model.eval()
    text_list = []
    labels_pred = []
    conf_pred = []
    ind = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=None
            )
        output_list = outputs[0].detach().cpu().numpy()
        output_prob = [str(np.max(soft * 100)) for soft in softmax(output_list, axis=1)]
        output = [np.argmax(out) for out in output_list]
        if config.BATCH_SIZE != 1:
            logging.debug(f"[INFO] Sample {ind + 1} - {ind + config.BATCH_SIZE} Done.")
        else:
            logging.debug(f"[INFO] Sample {ind + 1} Done.")
        ind += config.BATCH_SIZE
        text_list.extend(batch["input_text"])
        labels_pred.extend(output)
        conf_pred.extend(output_prob)
    return text_list, labels_pred, conf_pred


"""
def add_sentiments_to_json(predictions):
    json_data = None
    dict_lst = json_data[config.TranscriptionColumn]
    for ind, dict in enumerate(dict_lst):
        dict["sentiment"] = predictions[ind]
    return json_data
"""


def save_prediction(out_dict):
    """
    Save Predictions to the OUT.JSON file in results directory.

    """
    utils_tools.save_json(out_dict)
