from torch.utils.data import DataLoader

from . import config
from .ml_engine import MarkerBackbone, MarkerDataset, MarkerEngine
from .utils import custom_logging, utils_tools


def model_serve(json_data):
    """

    Args:
        :paramjson_data: The input JSON from the API request component.
        :return : The Marker Classifier serving component response for the api integration.

    """

    logger = custom_logging.get_logger()
    checkpoint, tokenizer = utils_tools.get_model_and_tokenizer(logger=logger)
    serve_dataset = MarkerDataset.from_json(json_data=[json_data], tokenizer=tokenizer)
    # Instantiate the Data-loader.
    serve_data_loader = DataLoader(
        serve_dataset, batch_size=config.params.get("serve").get("batch_size")
    )
    # Instantiate Model Creator Class, load the model architecture and the weights.
    backbone_model = MarkerBackbone(model=checkpoint, tokenizer=tokenizer)
    logger.info("Instance of MarkerClassifier is created")
    model_engine = MarkerEngine(
        model=backbone_model,
        data_loader=(serve_data_loader, None),
    )
    logger.info("Instance of MarkerClassifierEngine is created")
    response = model_engine.serve()
    status = response.get("marker_classifier_result").get("status")
    if status == "Success":
        marker_classifier_output = utils_tools.create_output_json(
            status=status,
            json_data=json_data,
            response=response.get("marker_classifier_result").get("details"),
        )
        utils_tools.save_result(json_data=marker_classifier_output)
        return marker_classifier_output
    else:
        json_data["status"] = status.lower()
        json_data["model"] = "marker_classifier"
        return json_data
