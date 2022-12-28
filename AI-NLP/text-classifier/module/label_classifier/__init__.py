import os
import sys

from torch.utils.data import DataLoader

from . import config
from .ml_engine import LabelBackbone, LabelClassifierEngine, LabelDataset
from .utils import custom_logging, utils_tools


def model_serve(json_data):
    """
    Args:
        :param(json) The input JSON from the API request component.

        :returns : The Label Classifier serving component response for the api integration.

    """
    logger = custom_logging.get_logger()
    serve_dataset = LabelDataset.from_json(
        json_data=[json_data], tokenizer=config.tokenizer
    )
    # Instantiate the Data-loader.
    serve_data_loader = DataLoader(
        serve_dataset, batch_size=config.params.get("serve").get("batch_size")
    )
    # Instantiate Model Creator Class, load the model architecture and the weights.
    backbone_model = LabelBackbone(model=config.model, tokenizer=config.tokenizer)
    logger.info("Instance of LabelClassification is created")
    model_engine = LabelClassifierEngine(
        model=backbone_model,
        data_loader=(serve_data_loader, None),
    )
    logger.info("Instance of LabelClassificationEngine is created")
    response = model_engine.serve()
    status = response.get("label_classification_result").get("status")
    if status == "Success":
        label_classification_output = utils_tools.create_output_json(
            status=status,
            json_data=json_data,
            response=response.get("label_classification_result").get("details"),
        )
        # To save the result in directory configured in config module
        # utils_tools.save_result(json_data=label_classification_output)
        return label_classification_output
    else:
        json_data["status"] = status.lower()
        json_data["model"] = "label_classifier"
        return json_data
