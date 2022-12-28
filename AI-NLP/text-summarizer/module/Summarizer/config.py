import functools
import logging
import os
from datetime import datetime

import transformers

# Summarizer Specs
SUMMARIZATION_MODEL = "t5-small"  # pretrained T5 model name
FINETUNED_MODEL = (
    "FineTunedSummarizer_18-21-05-10-2021"  # Model name for functional testing
)
BASE_FINETUNED_MODEL = "FineTunedSummarizer_18-21-05-10-2021"  # Base fine tuned model
PARAPHRASING_MODEL = "tuner007/pegasus_paraphrase"  # Pegasus paraphrasing model name
OUT_JSON = datetime.now().strftime("output_%H-%M-%d-%m-%Y.json")
TEST_JSON = os.path.abspath(
    "Summarizer/dataset/test.json"
)  # For batch Json put "batch_test.json" here


USE_EFS = True
# Directory Paths
if USE_EFS:
    EFS_PATH = os.environ.get("efsMountCtrPath")
    EFS_ROOT_DIR = os.sep.join([EFS_PATH, "summarizer"])
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


if not USE_EFS and not os.path.exists(OUTPUT_LOG):
    os.makedirs(OUTPUT_LOG)
if not USE_EFS and not os.path.exists(OUTPUT_RESULTS):
    os.makedirs(OUTPUT_RESULTS)

# Initializes logger object
LOG_FILE = datetime.now().strftime("execution_log_%H-%M-%d-%m-%Y.log")
if not USE_EFS:
    LOG_FILE = datetime.now().strftime("log_%H-%M-%d-%m-%Y.log")
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

# Summarizer Model configuration
BATCH_SIZE = 4
INPUT_LENGTH = 512  # Max Input text length
OUTPUT_LENGTH = 150  # Max output summary length
NUM_SENTENCES = 1  # Number of sentences to be returned by Pegasus Paraphraser Model
NUM_BEAMS = 4
TEMPERATURE = 1.5
NUM_EPOCHS = 10  # Number of Epochs
LEARNING_RATE = 2e-5
DEVICE = "cuda"  # cuda or cpu


# File and column names
TRAINING_DATA = "3rd_person_pov_train.csv"
summary_col = "summary"
text_col = "transcript"
response_column = "response"
chunk_id = "chunk_id"

# Loads model/tokenizer into memory
@functools.lru_cache(maxsize=None)
def load_model(model_path: str):

    global model
    global tokenizer
    global optimizer

    logging.info(f"[{BASE_FINETUNED_MODEL}] Loading Model..")
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_path)
    logging.info(f"[{BASE_FINETUNED_MODEL}] Model loaded successfully")

    logging.info(f"[{BASE_FINETUNED_MODEL}] Loading Tokenizer..")
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_path)
    logging.info(f"[{BASE_FINETUNED_MODEL}] Tokenizer loaded successfully")

    logging.info(f"[{BASE_FINETUNED_MODEL}] Loading Optimizer..")
    optimizer = transformers.AdamW(model.parameters(), lr=LEARNING_RATE)
    logging.info(f"[{BASE_FINETUNED_MODEL}] Optimizer loaded successfully")


# To load the model/tokenizer into memory
load_model(model_path=os.sep.join([MODEL_DIR, BASE_FINETUNED_MODEL]))
