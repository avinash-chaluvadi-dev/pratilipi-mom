import os
import sys

from torch.utils.data import DataLoader

from . import config
from .ml_engine.data_loader import ClassifierDataset
from .ml_engine.engine import SentimentEngine
from .ml_engine.model import SentimentBackbone


# Model Serving Component for the integration.
def model_serve(test_input):
    """

    Args:
        test_input: The input JSON from the API request component.

    Returns: The sentiment serving component response for the api integration.

    """
    backbone_model = SentimentBackbone(config.BASE_FINETUNED_CLASSIFIER)
    dataset = ClassifierDataset(backbone_model, load_mode="serve", json_data=test_input)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    model_engine = SentimentEngine(model=backbone_model, data_loader=data_loader)
    response = model_engine.serve()

    if response["status"] == "Success":
        test_input["status"] = response["status"].lower()
        cls_output = response["response"]["classifier_output"]
        pred_processed = response["response"]["confidence_score"]
        for chunk, cls, conf in zip(test_input["response"], cls_output, pred_processed):
            chunk.update({"classifier_output": cls, "confidence_score": conf})
        test_input["model"] = "sentiment_classifier"
        return test_input
    else:
        test_input["status"] = response[status].lower()
        test_input["model"] = "sentiment_classifier"
        return test_input
