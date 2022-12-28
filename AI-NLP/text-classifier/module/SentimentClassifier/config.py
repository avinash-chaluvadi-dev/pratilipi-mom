import functools
import logging
import os
from datetime import datetime

import transformers

# Network Loading
# If user wants to train the model from scratch using the pretrained bert-base-uncased.
TRANSFORMER_MODEL = "bert-base-uncased"
BASE_FINETUNED_CLASSIFIER = "FineTunedClassifier_18-21-05-10-2021"
BASE_FINETUNED_SENTIMENT = "FineTunedSentimentClassifier_18-21-05-10-2021"
SAVE_FINETUNED_CLASSIFIER = datetime.now().strftime(
    "FineTunedClassifier_%H-%M-%d-%m-%Y/"
)


USE_EFS = True
# Directory Paths.
if USE_EFS:
    EFS_PATH = os.environ.get("efsMountCtrPath")
    EFS_ROOT_DIR = os.sep.join([EFS_PATH, "sentiment-classifier"])
    MODEL_DIR = os.sep.join([EFS_ROOT_DIR, "models"])
    # Pointing the ROOT_DIR to label_classifier root directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    LOAD_DIR = os.sep.join([ROOT_DIR, "dataset"])
    OUTPUT_LOG = os.sep.join([ROOT_DIR, "results", "run_logs"])
    OUTPUT_RUN = os.sep.join([ROOT_DIR, "results", "test_outputs"])
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    MODEL_DIR = os.sep.join([ROOT_DIR, "models"])
    LOAD_DIR = os.sep.join([ROOT_DIR, "dataset"])
    OUTPUT_LOG = os.sep.join([ROOT_DIR, "results", "run_logs"])
    OUTPUT_RUN = os.sep.join([ROOT_DIR, "results", "test_outputs"])
LOG_FILE = datetime.now().strftime("execution_%H-%M-%d-%m-%Y.log")
OUT_JSON = datetime.now().strftime("output_%H-%M-%d-%m-%Y.json")

# Create the results directory's subdirectories.
if not USE_EFS and not os.path.exists(OUTPUT_LOG):
    os.makedirs(OUTPUT_LOG)
if not USE_EFS and not os.path.exists(OUTPUT_RUN):
    os.makedirs(OUTPUT_RUN)

if not USE_EFS:
    logging.basicConfig(
        filename=os.path.join(OUTPUT_LOG, LOG_FILE),
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )

# Data Artifact JSONs.
TEST_JSON = "test-inference.json"
TRAIN_JSON = "training_data.json"
EVAL_JSON = "eval.json"

# Network Parameters.
DEVICE = "cpu"


NUM_LABELS = 3
DECODE_LABELS = {1: "neutral", 0: "negative", 2: "positive"}

# JSON Keys for the inference
SERVE_TranscriptionColumn = "response"

# JSON keys for training
TRAIN_TranscriptionColumn = "train_data"
ClassifierColumn = "label"
TextColumn = "transcript"


BATCH_SIZE = 32
MAX_LENGTH = 512
EPOCHS = 1

# Serving Test Input
SERVE_INPUT = {
    "response": [
        {
            "transcript": "Okay so uh this is how the e-time tool looks like. Okay. So if you're a manager,"
            " you'll also get this kind of thing only payroll administrator role. "
            "you will be getting manage by employees tab since I'm, I admin, I have lots of access to it "
            "so you might have not have those access okay. So let me  brief you what access you have and "
            "what as a manager you have to do and what as employee you have to do okay ",
            "startTime": "5968.0",
            "endTime": "22800.0",
            "speakerId": 1,
        }
    ]
}

# Loads model/tokenizer into memory
@functools.lru_cache(maxsize=None)
def load_model(model_path: str):

    global model
    global tokenizer
    global optimizer

    logging.info(f"[{BASE_FINETUNED_SENTIMENT}] Loading Model..")
    model = transformers.BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=True,
    )
    logging.info(f"[{BASE_FINETUNED_SENTIMENT}] Model loaded successfully..")

    logging.info(f"[{BASE_FINETUNED_SENTIMENT}] Loading Tokenizer..")
    tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
    logging.info(f"[{BASE_FINETUNED_SENTIMENT}] Tokenizer loaded successfully..")

    logging.info(f"[{BASE_FINETUNED_SENTIMENT}] Loading Optimizer..")
    optimizer = transformers.AdamW(model.parameters(), lr=5e-5)
    logging.info(f"[{BASE_FINETUNED_SENTIMENT}] Optimizer loaded successfully..")


# To load the model/tokenizer into memory
load_model(model_path=os.sep.join([MODEL_DIR, BASE_FINETUNED_CLASSIFIER]))
