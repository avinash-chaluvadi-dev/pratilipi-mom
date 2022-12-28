import logging
import os
from typing import Dict, Optional, Tuple

from .. import config
from ..utils import utils_tools

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


class FramifyDataset:
    """
    FramifyDataset class - to load the dataset used the __getitem__ fashion supported by the Pytorch.
    The loader supports the JSON format for parsing the input to the network.

    Attributes:
        json_data: Input JSON data for serving
    """

    def __init__(self, json_data: Optional[Dict] = None) -> None:
        # Using the default JSON, when JSON data is not provided

        if json_data is None:
            json_data = utils_tools.load_json()

        self.json_data = json_data
        # Extracting video snippet paths, chunk ids from the JSON
        self.video_paths, self.chunk_ids = self._extract_data_from_json(json_data)

    def _extract_data_from_json(self, json_data: Optional[Dict]) -> Tuple[list, list]:
        """
        Extracting data from JSON

        Parameters:
            json_data (dict): JSON for the extraction of data.

        Returns:
            Tuple consisting of video snippet paths, chunk ids.

        """
        video_paths = []
        keys_list = json_data[config.response_key]
        ids = []
        for key in keys_list:
            path = key[config.video_path]
            chunk_id = key[config.chunk_id]
            ids.append(chunk_id)
            video_paths.append(path)
        return video_paths, ids

    def get_index_item(self, index: int):
        """Returns data dictionary for the given index"""

        return self.__getitem__(index)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        chunk_id = self.chunk_ids[index]
        return {
            "video_path": video_path,
            "chunk_id": chunk_id,
        }

    def __len__(self):
        return len(self.video_paths)
