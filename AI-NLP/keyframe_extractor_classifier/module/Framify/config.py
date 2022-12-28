import logging
import os
from datetime import datetime

USE_S3 = True
USE_EFS = True
S3_URI = "s3://"
IMAGE_CONTENT_TYPE = "image/jpeg"
AWS_STORAGE_BUCKET_NAME = "mom-bucket-uat"

# Directory Paths
if USE_EFS:
    EFS_PATH = os.environ.get("efsMountCtrPath")
    EFS_ROOT_DIR = os.sep.join([EFS_PATH, "keyframe-extraction"])
    MODEL_DIR = os.sep.join([EFS_ROOT_DIR, "models"])
    # Pointing the ROOT_DIR to keyframe_extraction root directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    INPUT_DIR = os.sep.join([ROOT_DIR, "dataset"])
    OUTPUT_LOG = os.sep.join(
        [ROOT_DIR, "results", "run_logs"]
    )  # Output directory for saving Logs
    OUTPUT_RESULTS = os.sep.join(
        [ROOT_DIR, "results", "test_outputs"]
    )  # Output directory for saving output JSONs
    KEYFRAMES_DIR = os.sep.join([OUTPUT_RESULTS, "Keyframes"])
    DEFAULT_GROUND_TRUTH = os.sep.join([INPUT_DIR, "Default Ground Truth"])
    DEFAULT_EVAL_VIDEO = os.sep.join([INPUT_DIR, "Default Evaluation Video.mp4"])
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    INPUT_DIR = os.sep.join([ROOT_DIR, "dataset"])
    MODEL_DIR = os.sep.join([BASE_DIR, "models"])
    OUTPUT_LOG = os.sep.join(
        [ROOT_DIR, "results", "run_logs"]
    )  # Output directory for saving Logs
    OUTPUT_RESULTS = os.sep.join(
        [ROOT_DIR, "results", "test_outputs"]
    )  # Output directory for saving output JSONs
    KEYFRAMES_DIR = os.sep.join([OUTPUT_RESULTS, "Keyframes"])
    DEFAULT_GROUND_TRUTH = os.sep.join([INPUT_DIR, "Default Ground Truth"])
    DEFAULT_EVAL_VIDEO = os.sep.join([INPUT_DIR, "Default Evaluation Video.mp4"])

if not USE_EFS and not os.path.exists(OUTPUT_LOG):
    os.makedirs(OUTPUT_LOG)
if not USE_EFS and not os.path.exists(OUTPUT_RESULTS):
    os.makedirs(OUTPUT_RESULTS)

TEST_JSON = "e6c1bca3-e7e6-4948-beb9-c53aaf54cfe4/speaker_diarization/response.json"  # For batch Json put "batch_test.json" here
OUT_JSON = datetime.now().strftime(
    "output_%H-%M-%d-%m-%Y.json"
)  # Output Json saved locally
LOG_FILE = datetime.now().strftime("log_%H-%M-%d-%m-%Y.log")
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

GROUND_TRUTH = None  # Ground truth directory path
EVAL_VIDEO = None  # Evaluation video

# Key values in JSON
keyframe_key = "keyframes"
video_path = "video_path"
chunk_id = "speaker_label"
response_key = "response"

# If both of the below are False, Absolute difference will be used
HASHING = False  # Whether to use Image hashing or not
DO_OPTICAL_FLOW = True  # Whether to use optical flow algorithm or not

# Network Parameters
THRESHOLD = {"absolute_difference": 200000, "hashing": 37, "optical_flow": 0.42}
USE_THRESHOLD = True  # Whether to use threshold based saving of frames (False means to use local maxima)
CHANGE_FPS = True  # Whether to change the default FPS
FRAME_RATE = 1 if DO_OPTICAL_FLOW is True else 1 / 2  # 1/ FPS
HEIGHT, WIDTH = 920, 1820  # Height and Width for Cropped Images
rWidth, rHeight = 720, 480  # Height and width for resizing the image
dsize = (rWidth, rHeight)
SMOOTHENING_WINDOW_LIST = [
    "flat",
    "hanning",
    "hamming",
    "bartlett",
    "blackman",
]  # Used in smoothening the array
DEFAULT_WINDOW_TYPE = SMOOTHENING_WINDOW_LIST[1]
FRAME_FORMAT = "jpg"  # jpg or png

NUM_INFO = 100  # Total number of information to be shown in the log file

UI_RENDER = True  # Boolean value to render keyframes to UI


def set_value(output_location):
    """
    Setting the keyframes output location
    Parameters:
        output_location: Path for saving the extracted keyframes

    """
    global KEYFRAMES_DIR
    KEYFRAMES_DIR = output_location
    logging.info(f"Output Directory for Keyframes changed to : {KEYFRAMES_DIR}")
