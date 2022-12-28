import functools
import os
from datetime import datetime

import spacy

from .utils import custom_logging

N_ITER = 1
DROPOUT = 0.5
USE_EFS = True

# Directory Paths
if USE_EFS:
    EFS_PATH = os.environ.get("efsMountCtrPath")
    EFS_ROOT_DIR = os.sep.join([EFS_PATH, "spacy-ner"])
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

JSONL_DIR = os.path.join(LOAD_DIR, "jsonl data")
JSONL_COLUMNS = ["data", "label"]


MODEL_NAME_LOAD = None
MODEL_NAME_SAVE = datetime.now().strftime("NER_Model_%H-%M-%d-%m-%Y")
BEST_MODEL = os.path.join(MODELS_DIR, "NER_Model_16-31-01-09-2021")
BASE_FINETUNED_NER = "FinetunedNER_16-31-01-09-2021"
OUT_JSON = datetime.now().strftime("output_%H-%M-%d-%m-%Y.json")
CONFIDENCE_THRESHOLD = 0.1

params = {
    "train": {
        "supported_dataset_type": ["csv", "json"],
        "dataset_files": ["NER_train.csv"],
        "package_test_files": ["ner.json", "batch_test.json"],
        "package_test_epochs": 1,
        "response_key": "response",
        "text_key": "transcript",
        "entities_key": "entities",
    },
    "eval": {
        "supported_dataset_type": ["csv", "json"],
        "dataset_files": ["NER_test.csv", "ner.json", "batch_test.json"],
        "package_test_files": ["ner.json", "batch_test.json"],
        "response_key": "response",
        "text_key": "transcript",
        "entities_key": "entities",
    },
    "serve": {
        "supported_dataset_type": ["json"],
        "dataset_files": ["ner.json", "batch_test.json"],
        "package_test_files": ["ner.json", "batch_test.json"],
        "response_key": "response",
        "text_key": "transcript",
        "entities_key": "entities",
        "chunk_id_key": "chunk_id",
        "confidence_score_key": "confidence_score",
    },
}

# Loads model/tokenizer into memory
@functools.lru_cache(maxsize=None)
def load_model(model_path: str):

    # Initializes logger object
    logger = custom_logging.get_logger()

    global ner
    global model

    logger.info(f"[{BASE_FINETUNED_NER}] Loading Model..")
    model = spacy.load(model_path)
    logger.info(f"[{BASE_FINETUNED_NER}] Model loaded successfully..")

    logger.info(f"[{BASE_FINETUNED_NER}] Loading NER Pipeline..")
    ner = model.get_pipe("ner")
    logger.info(f"[{BASE_FINETUNED_NER}] NER Pipeline loaded successfully")


# To load the model/tokenizer into memory
load_model(model_path=BEST_MODEL)
