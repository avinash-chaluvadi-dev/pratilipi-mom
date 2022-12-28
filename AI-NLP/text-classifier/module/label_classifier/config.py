import functools
import os
from datetime import datetime

from transformers import AdamW

from .utils import custom_logging, utils_tools

# Label classifier params
BASE_FINETUNED_MODEL = "FineTunedClassifier_23-04-25-01-2022"
BASE_FINETUNED_LABEL = "FineTunedLabelClassifier_23-04-25-01-2022"
OUT_JSON = datetime.now().strftime("output_%H-%M-%d-%m-%Y.json")
LABEL_CLASSIFIER = datetime.now().strftime("FineTunedClassifier_%H-%M-%d-%m-%Y")
LABEL_CLASSIFIER_BIN = "label_classifier.bin"


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

# Tran/Eval/Serve config params
MAX_LEN = 512
NUM_FOLDS = 5
NUM_EPOCHS = 1
TARGET_COLUMNS = ["label"]
BERT_MODEL = "bert-base-uncased"
CLASSIFICATION_LABELS = [
    "Action with Deadline",
    "Announcement",
    "Appreciation",
    "Action",
    "Others",
]
params = {
    "csv": {"text_column": "text", "label_column": "labels"},
    "json": {
        "response_key": "response",
        "text_key": "transcript",
        "label_key": "labels",
    },
    "train": {
        "batch_size": 1,
        "dataset_type": "csv",
        "dataset_files": ["labels.csv"],
        "model": "FineTunedClassifier_23-04-25-01-2022",
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
        "dataset_files": ["serve1.json"],
    },
    "package_test": {
        "package_test_epochs": 1,
        "dataset_type": "json",
        "dataset_files": ["train.json"],
    },
}


# Loads model/tokenizer into memory
@functools.lru_cache(maxsize=None)
def load_model(model_path: str = None):

    # Initializes logger object
    logger = custom_logging.get_logger()

    global model
    global tokenizer
    global optimizer

    logger.info(f"[{BASE_FINETUNED_LABEL}] Loading Model..")
    logger.info(f"[{BASE_FINETUNED_LABEL}] Loading Tokenizer..")

    model, tokenizer = utils_tools.get_model_and_tokenizer(logger=logger)
    logger.info(f"[{BASE_FINETUNED_LABEL}] Model loaded successfully..")
    logger.info(f"[{BASE_FINETUNED_LABEL}] Tokenizer loaded successfully..")

    logger.info(f"[{BASE_FINETUNED_LABEL}] Loading Optimizer..")
    optimizer = AdamW(model.parameters(), lr=3e-5)
    logger.info(f"[{BASE_FINETUNED_LABEL}] Optimizer loaded successfully..")


# To load the model/tokenizer into memory
load_model()
