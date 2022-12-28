import os
from datetime import datetime

USE_EFS = True

# Directory Paths.
if USE_EFS:
    EFS_PATH = os.environ.get("efsMountCtrPath")
    EFS_ROOT_DIR = os.sep.join([EFS_PATH, "feedback-adapter"])
    # Pointing the ROOT_DIR to label_classifier root directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    DATA_DIR = ROOT_DIR + "/data"
    STORE_DIR = ROOT_DIR + "/results"

else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    DATA_DIR = ROOT_DIR + "/data"
    STORE_DIR = ROOT_DIR + "/results"


if not os.path.exists(STORE_DIR):
    os.mkdir(STORE_DIR)

LABEL_DICT = DATA_DIR + "/labels.json"
RULES_DICT = DATA_DIR + "/rules.json"
TEST_DICT = DATA_DIR + "/test.json"
RESULT_STORE = STORE_DIR + datetime.now().strftime("/output_%H-%M-%d-%m-%Y.json")
LOG_STORE = STORE_DIR + datetime.now().strftime("/execution_%H-%M-%d-%m-%Y.log")

# This needs to be fetched from the global config store -> speaker diarizer.
NUM_SPEAKER = 4
