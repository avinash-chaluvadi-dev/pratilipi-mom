import os
from datetime import datetime

USE_EFS = True

# Directory Paths
if USE_EFS:
    EFS_PATH = os.environ.get("efsMountCtrPath")
    EFS_ROOT_DIR = os.sep.join([EFS_PATH, "label-classifier"])
    MODELS_DIR = os.sep.join([EFS_ROOT_DIR, "models"])
    # Pointing the ROOT_DIR to label_classifier root directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    LOAD_DIR = os.sep.join([ROOT_DIR, "dataset"])
    OUTPUT_LOG = os.sep.join([ROOT_DIR, "results", "run_logs"])
    OUTPUT_RUN = os.sep.join([ROOT_DIR, "results", "test_outputs"])
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    MODELS_DIR = os.sep.join([ROOT_DIR, "models"])
    LOAD_DIR = os.sep.join([ROOT_DIR, "dataset"])
    OUTPUT_LOG = os.sep.join([ROOT_DIR, "results", "run_logs"])
    OUTPUT_RUN = os.sep.join([ROOT_DIR, "results", "test_outputs"])

OUT_JSON = datetime.now().strftime("output_%H-%M-%d-%m-%Y.json")
MARKER_CLASSIFIER = datetime.now().strftime("FineTunedClassifier_%H-%M-%d-%m-%Y")
MARKER_CLASSIFIER_BIN = "marker_classifier.bin"

MAX_LEN = 512
NUM_FOLDS = 5
NUM_EPOCHS = 10
TARGET_COLUMNS = ["marker"]
CLASSIFICATION_LABELS = [
    "Action Plan Tracking",
    "Proactiveness",
    "Collaboration",
    "Mentoring&Engagement",
    "Others",
]
BERT_MODEL = "bert-base-uncased"
params = {
    "csv": {"text_column": "text", "marker_column": "marker"},
    "json": {
        "response_key": "response",
        "text_key": "transcript",
        "marker_key": "marker",
    },
    "train": {
        "batch_size": 1,
        "dataset_type": "json",
        "dataset_files": ["train.json"],
    },
    "eval": {
        "batch_size": 4,
        "dataset_type": "json",
        "dataset_files": ["eval.json"],
        "shuffle": True,
    },
    "serve": {
        "batch_size": 4,
        "dataset_type": "json",
        "dataset_files": ["serve.json"],
    },
    "package_test": {
        "package_test_epochs": 1,
        "dataset_type": "json",
        "dataset_files": ["train.json"],
    },
}
