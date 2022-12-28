import os

from torch.utils.data.dataloader import DataLoader

from . import config
from .ml_engine.data_loader import HeadlinerDataset
from .ml_engine.engine import HeadlinerEngine
from .ml_engine.model import HeadlinerBackbone
from .utils import utils_tools


def model_serve(data):
    """
    Model serving function

    Parameters:
        data: Input texts/ JSON for prediction from the API request (format - JSON).

    Returns:
        Response from the Headliner serving component.

    """

    # Loading backbone model
    backbone_model = HeadlinerBackbone(model_name=config.BASE_FINETUNED_MODEL)

    # Initializing the Headliner Dataset
    dataset = HeadlinerDataset(model=backbone_model, json_data=data, is_train=False)

    # Initialize the Data-loader.
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Run the Headliner Engine
    headliner_engine = HeadlinerEngine(backbone_model, dataset, data_loader)

    # Getting the response
    response = headliner_engine.serve(save_result=False)
    return response
