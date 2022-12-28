import logging
import os
from pathlib import Path
from typing import Dict, Union

from .. import config
from ..utils import utils_tools
from .data_loader import FramifyDataset
from .evaluation import get_scores
from .model import FramifyBackbone

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


class FramifyEngine:
    """
    FramifyEngine Class - wrapper class for encapsulating the evaluate and serve function of the
    Framify package

    Attributes:
        model: FramifyBackbone model
        dataset: FramifyDataset for input data

    """

    def __init__(
        self, backbone_model: FramifyBackbone, dataset: FramifyDataset
    ) -> None:
        self.model = backbone_model
        self.dataset = dataset

    def train(self):
        raise NotImplementedError("Framify does not support training method currently.")

    def evaluate(
        self,
        input_file: Union[str, Path] = None,
        ground_truth_dir: Union[str, Path] = None,
    ) -> Dict[str, float]:
        """
        Evaluation method for Framify

        Parameters:
            input_file: input file path for the video snippet to be used for evaluation
            ground_truth_dir: directory path for the ground truth data for the input_file

        Returns:
            Dictionary of Evaluation Scores

        """
        # Using default evaluation video when input file path is not given
        if input_file is None:
            input_file = config.DEFAULT_EVAL_VIDEO
            ground_truth_dir = config.DEFAULT_GROUND_TRUTH

        if ground_truth_dir is None:
            logging.exception(
                "ground_truth_dir should not be None if input_file is passed"
            )
            raise NameError(
                "ground_truth_dir should not be None if input_file is passed"
            )

        file_paths = self.model.run(input_file)  # Getting paths of extracted frames
        prediction_dir = os.path.dirname(file_paths[0])  # Getting directory path
        return get_scores(prediction_dir, ground_truth_dir)

    def serve(self, save_result: bool = False):
        """
        Serving method for Framify (Keyframe extractor)

        Parameters:
            save_result: default False
                Whether or not to save the result locally (True for functional tests)

        Returns:
            The output JSON containing process status and details (paths of key frames).

        """
        try:
            logging.debug("Extracting Keyframes...")
            num_transcript = len(self.dataset)
            num_frames_list = []
            keyframe_paths = []
            json_data = self.dataset.json_data

            # Iterating over each video snippets and storing important info
            for data in self.dataset:
                path = data.get("video_path")
                chunk_id = data.get("chunk_id")

                # getting saved frame paths
                print(
                    f"path is {path} and chunk_id is {chunk_id} and num_trans is {num_transcript}"
                )
                file_paths = self.model.run(path, chunk_id, num_transcript)
                num_frames_list.append(len(file_paths))
                keyframe_paths.append(file_paths)

            response = utils_tools.get_response(
                keyframe_paths, json_data=json_data, status="Success"
            )

            # Check if serving is run in the save mode, and store output locally.
            if save_result is True:
                logging.info("Saving Results locally")
                utils_tools.save_prediction(response)

        except Exception:
            response = utils_tools.get_response(
                None, json_data=self.dataset.json_data, status="Error"
            )
            logging.exception(
                "Exception occurred while serving the Framify(Keyframe Extraction) model",
                exc_info=True,
            )

        # return utils_tools.dict_to_json(response)
        return response
