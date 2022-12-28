"""data_loader module
This script allows us to create text file
corresponding to each meeting
"""

import os
import pathlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import config
from ..utils import custom_logging, utils_tools


class DatasetCreator:
    def __init__(self, frames_dir: str = None, video_dir: str = None):
        self.video_dir = video_dir
        self.frames_dir = frames_dir
        self.logger = custom_logging.get_logger()
        self.input_dir = os.path.normpath(os.path.abspath(config.LOAD_DIR))

    def get_frames(self, video_filename, seconds, count):
        video_cap = cv2.VideoCapture(video_filename)
        video_cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        success, image = video_cap.read()
        if success:
            dir_name = os.path.basename(os.path.splitext(video_filename)[0])
            dir_path = os.path.join(self.frames_dir, dir_name)
            if not utils_tools.path_exists(path=dir_path):
                os.makedirs(dir_path)
            cv2.imwrite(os.path.join(dir_path, f"{dir_name}_{count}.jpg"), image)
        return success

    def video_to_frames(self, video_filename=None):
        try:
            if not utils_tools.path_exists(self.frames_dir):
                os.makedirs(self.frames_dir)
            seconds = 0
            count = 1
            if video_filename:
                while self.get_frames(
                    video_filename=os.path.abspath(video_filename),
                    seconds=seconds,
                    count=count,
                ):
                    seconds = seconds + config.FRAME_RATE
                    count = count + 1
            else:
                for video_filename in os.listdir(self.video_dir):
                    seconds = 0
                    count = 1
                    while self.get_frames(
                        video_filename=os.path.join(self.video_dir, video_filename),
                        seconds=seconds,
                        count=count,
                    ):
                        seconds = seconds + config.FRAME_RATE
                        count = count + 1
        except MemoryError:
            self.logger.exception("Memory error occurred, ran out of memory")
        except (RuntimeError, ValueError, TypeError):
            self.logger.exception("Run Time Exception Occurred")

    def create_train_data(self):
        try:
            x, y = [], []
            data_dir = pathlib.Path(self.frames_dir)
            for images_dir in os.listdir(str(data_dir)):
                x = x + [
                    str(img_path).replace("\\", "/")
                    for img_path in list(data_dir.glob(f"{images_dir}/*"))
                ]
                y = y + [images_dir] * len(list(data_dir.glob(f"{images_dir}/*")))
            dataframe = pd.DataFrame(
                {
                    config.INPUT_COLUMN: x,
                    config.TARGET_COLUMNS[0]: y,
                    "kf": [-1] * len(x),
                }
            )
            if not utils_tools.path_exists(
                path=os.path.join(self.input_dir, config.DATA_CSV)
            ):
                dataframe.to_csv(
                    os.path.join(self.input_dir, config.DATA_CSV), index=False
                )
            else:
                dataframe = pd.read_csv(os.path.join(self.input_dir, config.DATA_CSV))
            return dataframe
        except MemoryError:
            self.logger.exception("Memory error occurred, ran out of memory")
        except (RuntimeError, ValueError, TypeError):
            self.logger.exception("Run Time Exception Occurred")


class DatasetFormatter:
    """
    DatasetFormatter class - to format the items in dataset object
    """

    @staticmethod
    def image_to_array(image):
        """This function is used to convert image into pixel values.

        ARGS:
            image(str): Absolute Path of Images.

        """
        s3_utils = utils_tools.S3Utils(bucket_name=config.AWS_STORAGE_BUCKET_NAME)
        if config.USE_S3 == True:
            if not isinstance(image, list):
                image_bytes = s3_utils.load_data(
                    load_prefix=s3_utils.parse_s3_uri(file_uri=image)
                )
                return cv2.imdecode(
                    np.asarray(bytearray(image_bytes)), cv2.IMREAD_COLOR
                )
            else:
                image_array = []
                for img in image:
                    image_bytes = s3_utils.load_data(
                        load_prefix=s3_utils.parse_s3_uri(file_uri=image)
                    )
                    image_array.append(
                        cv2.imdecode(
                            np.asarray(bytearray(image_bytes)), cv2.IMREAD_COLOR
                        )
                    )
                return image_array
        else:
            if not isinstance(image, list):
                return cv2.imread(str(image))
            return [cv2.imread(str(img)) for img in image]

    @staticmethod
    def resize_rescale(image, framework):
        """This function is used to resize and rescale the image.

        ARGS:
            image(numpy.ndarray): array of pixel values.
            framework(str): can be pytorch, tensorflow, and keras.

        """
        resized_img = cv2.resize(image, config.RESIZE_SHAPE)
        if framework == "pytorch":
            resized_img = resized_img.reshape(
                resized_img.shape[2], resized_img.shape[0], -1
            )
        rescaled_img = resized_img / 255
        return rescaled_img


class ImageDataset(Dataset):
    """
    ImageDataset class - to load the dataset used the __getitem__ fashion supported by the Pytorch.
    The loader supports the JSON and the csv format for parsing the input to the network.
    Note: only CSV format is supported for the training, while CSV and JSON are supported for the evaluation
    and testing.
    """

    logger = custom_logging.get_logger()

    def __init__(
        self,
        images,
        labels=None,
        framework=None,
        mode=None,
        chunk_ids=None,
        speaker_ids=None,
        func_test=None,
    ):
        """This function is used to initialize the state of
        object(ImageDataset).

        ARGS:
            images(numpy.ndarray, list): Array of absolute paths of images.
            labels(numpy.ndarray, list): Array of image labels/classes.

        """
        self.mode = mode
        self.images = images
        self.labels = labels
        self.func_test = func_test
        self.framework = framework
        self.chunk_ids = chunk_ids
        self.speaker_ids = speaker_ids

    @classmethod
    def from_dataframe(cls, dataframe, mode, framework="pytorch"):
        images = dataframe.loc[:, config.INPUT_COLUMN].values
        labels = dataframe.loc[:, config.TARGET_COLUMNS[0]].values
        return cls(images=images, labels=labels, mode=mode, framework=framework)

    @classmethod
    def from_json(cls, json_data, mode="serve", func_test=None):
        images = []
        labels = []
        speaker_ids = []
        chunk_ids = []
        for keyframe in json_data.get(config.params.get("infer").get("response_key")):
            keyframes_path = keyframe.get(
                config.params.get("infer").get("keyframe_key")
            )
            images.extend(keyframes_path)
            if mode in ["train", "eval"]:
                labels.extend(keyframe.get("labels"))
            elif mode == "package_test" and func_test != "serve":
                labels.extend(keyframe.get("labels"))
            elif mode in ["package_test"] and func_test == "serve":
                speaker_ids.extend([keyframe.get("speaker_id")] * len(keyframes_path))
                chunk_ids.extend([keyframe.get("chunk_id")] * len(keyframes_path))
            else:
                speaker_ids.extend([keyframe.get("speaker_id")] * len(keyframes_path))
                chunk_ids.extend([keyframe.get("speaker_label")] * len(keyframes_path))

        if len(images) == 0:
            cls.logger.exception("No keyframes were present in Json..")
        else:
            return cls(
                images=images,
                labels=labels,
                framework="pytorch",
                mode=mode,
                speaker_ids=speaker_ids,
                chunk_ids=chunk_ids,
                func_test=func_test,
            )

    def __getitem__(self, item):
        image = self.images[item]
        speaker_id = self.speaker_ids[item]
        chunk_id = self.chunk_ids[item]
        if isinstance(image, str):
            image = DatasetFormatter.image_to_array(image)
        cleaned_image = DatasetFormatter.resize_rescale(image, framework=self.framework)
        if self.mode in ["train", "eval"]:
            label = self.labels[item]
            return {
                "image": torch.tensor(cleaned_image, dtype=torch.float),
                "label": torch.tensor(
                    config.CLASSIFICATION_LABELS.index(label), dtype=torch.long
                ),
            }
        elif self.mode == "package_test" and self.func_test != "serve":
            label = self.labels[item]
            return {
                "image": torch.tensor(cleaned_image, dtype=torch.float),
                "label": torch.tensor(
                    config.CLASSIFICATION_LABELS.index(label), dtype=torch.long
                ),
            }
        elif self.mode == "package_test" and self.func_test == "serve":
            return {
                "image": torch.tensor(cleaned_image, dtype=torch.float),
                "speaker_id": speaker_id,
                "chunk_id": torch.tensor(chunk_id, dtype=torch.long),
            }
        else:
            return {
                "image": torch.tensor(cleaned_image, dtype=torch.float),
                "speaker_id": speaker_id,
                "chunk_id": chunk_id,
            }

    def __len__(self):
        return len(self.images)
