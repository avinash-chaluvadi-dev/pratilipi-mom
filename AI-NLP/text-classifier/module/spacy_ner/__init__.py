from . import config
from .ml_engine import data_loader, engine, model
from .utils import custom_logging


def model_serve(json_data):
    """
    Args:
        :param(json) The input JSON from the API request component.

        :returns : The SpacyNER serving component response for the api integration.

    """
    logger = custom_logging.get_logger()
    serve_data_loader = data_loader.NERDataset.from_json(
        json_data=json_data, mode="serve"
    )
    # Instantiate Model Creator Class, load the model architecture and the weights.
    backbone_model = model.SpacyNER.from_spacy_model(config.BEST_MODEL)
    logger.info("Instance of SpacyNER is created")
    model_engine = engine.NEREngine(
        model=backbone_model, serve_data_loader=serve_data_loader.data
    )
    logger.info("Instance of NEREngine is created")
    response = model_engine.serve()
    status = response.get("ner_result").get("status")
    if status == "Success":
        return {
            "status": "success",
            "model": "Spacy_NER",
            "response": response.get("ner_result").get("details"),
        }
    else:
        json_data["status"] = status.lower()
        json_data["model"] = "Spacy_NER"
        return json_data
