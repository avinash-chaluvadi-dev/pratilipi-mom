import os
import sys
from typing import Dict, List, Union

import numpy as np
from torch.utils.data.dataloader import DataLoader

from . import config
from .ml_engine.data_loader import TransformerModelDataset
from .ml_engine.engine import AllocatorEngine
from .ml_engine.model import AllocatorBackbone
from .utils import utils_tools


def model_serve(data: Union[Dict, List]) -> Dict[str, Union[str, list]]:
    """
    Model serving function

    Parameters:
        data: Input texts/ JSON for prediction from the API request (format - list, JSON).

    Returns:
        Response from the Allocator (Deadline and Escalation classifier) serving component.

    """

    if isinstance(data, (list, np.ndarray)):
        json_data = utils_tools.list_to_json(text_list=data)
    else:
        json_data = data

    if config.SEQUENCE_CLASSIFICATION is True:
        model_name = config.BASE_FINETUNED_CLASSIFIER
    else:
        model_name = config.TRANSFORMER_MODEL

    # Loading the backbone model
    backbone_model = AllocatorBackbone(
        model_name=model_name,
        sequence_classification=config.SEQUENCE_CLASSIFICATION,
    )

    # Initializing the dataset
    dataset = TransformerModelDataset(
        backbone_model, json_data=json_data, is_train=False
    )

    # Initialize the Data-loader.
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Run the Allocator Engine.
    model_engine = AllocatorEngine(
        model=backbone_model,
        data_loader=data_loader,
        dataset=dataset,
        sequence_classification=config.SEQUENCE_CLASSIFICATION,
    )

    # Getting response from serve method
    response = model_engine.serve(save_result=False)
    return response
