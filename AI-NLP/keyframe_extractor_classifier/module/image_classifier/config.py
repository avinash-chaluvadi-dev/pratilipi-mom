import functools
import os
from datetime import datetime

import torch
from torch.optim import Adam

from .ml_engine import FrameClassifier
from .utils import custom_logging

USE_EFS = True

# S3 Config Params
USE_S3 = True
S3_URI = "s3://"
AWS_STORAGE_BUCKET_NAME = "mom-bucket-uat"

# Directory Paths
if USE_EFS:
    EFS_PATH = os.environ.get("efsMountCtrPath")
    EFS_ROOT_DIR = os.sep.join([EFS_PATH, "keyframe-classifier"])
    MODELS_DIR = os.sep.join([EFS_ROOT_DIR, "models"])
    # Pointing the ROOT_DIR to label_classifier root directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    LOAD_DIR = os.sep.join([ROOT_DIR, "dataset"])
    FRAMES_DIR = os.sep.join([LOAD_DIR, "images"])
    EVAL_FRAMES_DIR = os.sep.join([LOAD_DIR, "eval_images"])
    VIDEO_DIR = os.sep.join([LOAD_DIR, "video_recordings"])
    OUTPUT_LOG = os.sep.join([ROOT_DIR, "results", "run_logs"])
    OUTPUT_RUN = os.sep.join([ROOT_DIR, "results", "test_outputs"])
else:
    ROOT_DIR = os.sep.dirname(os.path.abspath(__file__))
    LOAD_DIR = os.sep.join([ROOT_DIR, "dataset"])
    MODELS_DIR = os.sep.join([ROOT_DIR, "models"])
    FRAMES_DIR = os.sep.join([LOAD_DIR, "images"])
    EVAL_FRAMES_DIR = os.sep.join([LOAD_DIR, "eval_images"])
    VIDEO_DIR = os.sep.join([LOAD_DIR, "video_recordings"])
    OUTPUT_LOG = os.sep.join([ROOT_DIR, "results", "run_logs_test"])
    OUTPUT_RUN = os.sep.join([ROOT_DIR, "results", "test_outputs"])

IMAGE_CLASSIFIER = datetime.now().strftime("FineTunedClassifier_%H-%M-%d-%m-%Y.pth")
OUT_JSON = datetime.now().strftime("output_%H-%M-%d-%m-%Y.json")
BEST_MODEL = "FineTunedClassifier_17-12-19-08-2021.pth"
KEYFRAME_CLASSIFIER = "FineTunedKeyframeClassifier_17-12-19-08-2021"
DATA_CSV = "data.csv"
EVALUATION_DATA = "eval.csv"

## Config Params
NUM_FOLDS = 5
NUM_EPOCHS = 20
FRAME_RATE = 0.5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
RESIZE_SHAPE = (128, 128)
INPUT_SHAPE = (1, 3, 128, 128)
CLASSIFICATION_LABELS = [
    "Confluence",
    "Excel",
    "Jira",
    "MS Teams",
    "MS Word",
    "OneNote",
    "Outlook",
    "PPT",
]
DEVICE = "cuda"
INPUT_COLUMN = "image_path"
TARGET_COLUMNS = ["image_label"]

params = {
    "train": {
        "batch_size": 8,
        "dataset_type": ["csv", "json"],
        "dataset_files": ["data.csv"],
        "package_test_files": ["keyframes.json"],
        "frame_rate": 0.5,
        "model": "FineTunedClassifier_17-12-19-08-2021.pth",
        "package_test_epochs": 1,
    },
    "eval": {
        "batch_size": 4,
        "dataset_type": "json",
        "dataset_files": ["keyframes.json"],
        "package_test_files": ["keyframes.json"],
        "shuffle": True,
        "accuracy": 0.82,
        "best_model": "FineTunedClassifier_17-12-19-08-2021.pth",
    },
    "infer": {
        "batch_size": 4,
        "dataset_type": "json",
        "package_test_files": ["keyframe_extraction_output.json"],
        "dataset_files": ["keyframe_extraction_output.json"],
        "transcription_key": "transcriptions",
        "response_key": "response",
        "keyframe_key": "keyframes",
    },
}

NETWORK = {
    "CNN": {
        "num_convolution_layers": 4,
        "in_channels": [INPUT_SHAPE[1], 32, 64, 128],
        "out_channels": [32, 64, 128, 128],
        "kernel_sizes": [(3, 3), (4, 4), (3, 3), (2, 2)],
        "stride": [(1, 1), (1, 1), (1, 1), (1, 1)],
        "padding": [0, 1, 1, 0],
        "num_dense_layers": 4,
        "dense_layer_input_output_features": [
            (None, 128),
            (128, 64),
            (64, 32),
            (32, len(CLASSIFICATION_LABELS)),
        ],
    },
}


@functools.lru_cache(maxsize=None)
def load_model(model_path: str):

    # Initializes logger object
    logger = custom_logging.get_logger()

    global model
    global optimizer

    backbone_model = FrameClassifier()
    if torch.cuda.is_available() and DEVICE == "cuda":
        logger.info(f"[{KEYFRAME_CLASSIFIER}] Loading Model..")
        model = torch.load(model_path)
        logger.info(f"[{KEYFRAME_CLASSIFIER}] Model loaded successfully..")

        logger.info(f"[{KEYFRAME_CLASSIFIER}] Loading Optimizer..")
        optimizer = Adam(backbone_model.parameters(), lr=0.001, weight_decay=0.0001)
        logger.info(f"[{KEYFRAME_CLASSIFIER}] Optimizer loaded successfully..")

    else:
        logger.info(f"[{KEYFRAME_CLASSIFIER}] Loading Model..")
        model = torch.load(model_path, map_location="cpu")
        logger.info(f"[{KEYFRAME_CLASSIFIER}] Model loaded successfully..")

        logger.info(f"[{KEYFRAME_CLASSIFIER}] Loading Optimizer..")
        optimizer = Adam(backbone_model.parameters(), lr=0.001, weight_decay=0.0001)
        logger.info(f"[{KEYFRAME_CLASSIFIER}] Optimizer loaded successfully..")


# To load the model/tokenizer into memory
load_model(model_path=os.sep.join([MODELS_DIR, params["eval"]["best_model"]]))
