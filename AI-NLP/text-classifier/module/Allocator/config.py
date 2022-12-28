import os
from datetime import datetime

USE_EFS = True

# Directory Paths.
if USE_EFS:
    EFS_PATH = os.environ.get("efsMountCtrPath")
    EFS_ROOT_DIR = os.sep.join([EFS_PATH, "sentiment-classifier"])
    MODEL_DIR = os.sep.join([EFS_ROOT_DIR, "models"])
    # Pointing the ROOT_DIR to label_classifier root directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    INPUT_DIR = os.sep.join([ROOT_DIR, "dataset"])
    OUTPUT_LOG = os.sep.join(
        [ROOT_DIR, "results", "run_logs"]
    )  # Output directory for saving Logs
    OUTPUT_RESULTS = os.sep.join(
        [ROOT_DIR, "results", "test_outputs"]
    )  # Output directory for saving output JSONs
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    MODEL_DIR = os.sep.join([ROOT_DIR, "models"])
    INPUT_DIR = os.sep.join([ROOT_DIR, "dataset"])
    OUTPUT_LOG = os.sep.join(
        [ROOT_DIR, "results", "run_logs"]
    )  # Output directory for saving Logs
    OUTPUT_RESULTS = os.sep.join(
        [ROOT_DIR, "results", "test_outputs"]
    )  # Output directory for saving output JSONs
LOG_FILE = datetime.now().strftime("execution_%H-%M-%d-%m-%Y.log")
OUT_JSON = datetime.now().strftime("output_%H-%M-%d-%m-%Y.json")
TEST_JSON = "test.json"  # For batch Json put "batch_test.json" here


if not USE_EFS and not os.path.exists(OUTPUT_LOG):
    os.makedirs(OUTPUT_LOG)
if not USE_EFS and not os.path.exists(OUTPUT_RESULTS):
    os.makedirs(OUTPUT_RESULTS)

# Keys related to JSON
response_column = "response"
allocation_col = "classifier_output"
chunk_id = "chunk_id"

# File and column names
TRAINING_DATA = "train.csv"
label_col = "labels"
text_col = "transcript"
ML_MODEL = "ml_model.pickle"
BASE_ML_MODEL = "base_ml_model.pickle"
BASE_FINETUNED_CLASSIFIER = "Finetuned Classifier"
FINETUNED_CLASSIFIER = "Testing Finetuned Classifier"

# Model Configuration
MAX_LEN = 224  # Maximum length of tokens to be passed to transformer models
BATCH_SIZE = 8  # Number of batches of data returned from Data loader/ batcher
NUM_EPOCHS = 1  # Number of epochs for training purposes
LEARNING_RATE = 2e-5  # Learning rate for optimizer
TRANSFORMER_MODEL_LIST = [
    "bert-base-uncased",
    "distilbert-base-uncased",
]
TRANSFORMER_MODEL = TRANSFORMER_MODEL_LIST[1]
PREPROCESS_TEXT = True  # whether to preprocess the input texts or not
DEVICE = "cuda"  # cuda or cpu

ML_CLF_MODEL_LIST = [
    "SGD Classifier",
    "Random Forest Classifier",
    "SVM Classifier",
    "XGBoost Classifier",
]
ML_CLF_MODEL = ML_CLF_MODEL_LIST[0]

# Global Random Seed
RANDOM_STATE = 100

LABEL_DICT = {
    "None": 0,
    "Deadline": 1,
    "Escalation": 2,
    "none": 0,
    "deadline": 1,
    "escalation": 2,
}
INVERSE_LABEL_DICT = {0: "None", 1: "Deadline", 2: "Escalation"}

NUM_LABELS = 3  # Number of labels
N_SPLITS = 5  # Number of splits in K fold
KFOLD_TRAINING = False  # Training using K fold validation
KFOLD_VALIDATION = False  # Whether to use K Fold validation for evaluation
SEQUENCE_CLASSIFICATION = True  # Whether or not to use Transformer for sequence classification else ML model with
# features from transformer model will be used
