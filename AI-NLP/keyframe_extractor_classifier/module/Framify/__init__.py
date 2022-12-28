import os
import sys
from pathlib import Path
from typing import Dict, Union

from . import config
from .ml_engine.data_loader import FramifyDataset
from .ml_engine.engine import FramifyEngine
from .ml_engine.model import FramifyBackbone

# sys.path.insert(1, os.getcwd())


def model_serve(data: Dict, keyframe_output_path: Union[Path, str] = None):
    """
    Model serving function

    Parameters:
        data: Input texts/ JSON for prediction from the API request (format JSON).
        keyframe_output_path: output path for saving the keyframes which are extracted

    Returns:
        Response from the Framify(Keyframe Extractor) serving component.

    """

    if keyframe_output_path is not None:
        # Changing the Keyframes output path
        config.set_value(keyframe_output_path)
    # Loading backbone model
    backbone_model = FramifyBackbone()

    # Initializing the Framify Dataset
    dataset = FramifyDataset(json_data=data)

    # Run the Framify Engine
    framify_engine = FramifyEngine(backbone_model, dataset)

    # Getting the response
    response = framify_engine.serve(save_result=False)

    return response
