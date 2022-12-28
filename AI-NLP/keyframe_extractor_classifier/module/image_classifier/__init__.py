from torch.utils.data import DataLoader

from . import config
from .ml_engine import FrameClassifier, FrameClassifierEngine, data_loader
from .utils import custom_logging, utils_tools


def model_serve(json_data, save_result=False):
    """
    Args:
        :param(json) The input JSON from the API request component.

        :returns : The Frame Classifier serving component response for the api integration.

    """
    logger = custom_logging.get_logger()
    dataset = data_loader.ImageDataset.from_json(json_data=json_data)
    # Instantiate the Data-loader.
    serve_data_loader = DataLoader(
        dataset, batch_size=config.params.get("infer").get("batch_size")
    )
    # Instantiate Model Creator Class, load the model architecture and the weights.
    backbone_model = FrameClassifier()
    logger.info("Instance of FrameClassifier is created")
    model_engine = FrameClassifierEngine(
        model=backbone_model,
        serve_data_loader=serve_data_loader,
    )
    logger.info("Instance of FrameClassifierEngine is created")
    response = model_engine.serve()
    status = response.get("keyframe_classification_result").get("status")
    if status == "Success":
        keyframe_classifier_output = utils_tools.create_output_json(
            status=status,
            json_data=json_data,
            response=response.get("keyframe_classification_result").get("details"),
        )
        if save_result:
            utils_tools.save_result(json_data=keyframe_classifier_output)
        return keyframe_classifier_output
    else:
        json_data["status"] = status.lower()
        json_data["model"] = "keyframe_classifier"
        return json_data
