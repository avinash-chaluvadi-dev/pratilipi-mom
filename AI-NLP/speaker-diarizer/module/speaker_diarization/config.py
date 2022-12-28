import os
from datetime import datetime

# DIRECTORY SETUP
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
DATA_DIR = ROOT_DIR + "/dataset"
SEG_RESULTS_DIR = ROOT_DIR + "/results/store_segs"
LOG_RESULTS_DIR = ROOT_DIR + "/results/store_logs"
LOG_FILE = datetime.now().strftime("/execution_%H-%M-%d-%m-%Y.log")
UISRNN_MODEL_STORE = ROOT_DIR + "/models/uisrnn"
VGG_VOX_MODEL_STORE = ROOT_DIR + "/models/vggvox"
TEST_FILE = DATA_DIR + "/kt_packaging_convention.wav"
GT_CSV = DATA_DIR + "/gt_csv.csv"

# Traversing and creating the above paths.
for direc in [
    DATA_DIR,
    SEG_RESULTS_DIR,
    LOG_RESULTS_DIR,
    UISRNN_MODEL_STORE,
    VGG_VOX_MODEL_STORE,
]:
    if not os.path.exists(direc):
        os.makedirs(direc)

# Engine run mode, to be used when the run in the pluggable mode.
RUN_MODE = "infer"

# DIARIZATION SERVING PARAMETERS.
SERVE = {
    "DEVICE": "cpu",
    "VGG_MODEL_PATH": VGG_VOX_MODEL_STORE + "/weights.h5",
    "UISRNN_MODEL_PATH": UISRNN_MODEL_STORE + "/saved_model.uisrnn_benchmark",
    "VGG_DATA_DIR": "4persons",
    # Choices = ['resnet34s', 'resnet34l'].
    "NET": "resnet34s",
    "GHOST_CLUSTER": 2,
    "VLAD_CLUSTER": 8,
    "BOTTLENECK_DIM": 512,
    # Choices=['avg', 'vlad', 'gvlad'].
    "AGGREGATION_MODE": "gvlad",
    # Choices=['softmax', 'amsoftmax'].
    "LOSS": "softmax",
    # Choices=['normal', 'hard', 'extend']
    "TEST_TYPE": "normal",
    "SAMPLE_PARAMS": {
        "DIM": (257, 0, 1),
        "N_FFT": 512,
        "SPEC_LEN": 250,
        "WIN_LENGTH": 400,
        "HOP_LENGTH": 160,
        "N_CLASSES": 5994,  # AN upper bound on the number of speakers
        "SAMPLING_RATE": 16000,
        "NORMALIZE": True,
    },
    "DIFF_DURATION": 2.0,
    "CUT_TH_DURATION": 2.0,
    "EMBEDDING_PER_SECOND": 1.2,
    "OVERLAP_RATE": 0.4,
}

# UISRNN DIARIZATION Parameters.
# Reference Paper link Fully Supervised Speaker Diarization - https://arxiv.org/pdf/1810.04719.pdf
UISRNN = {
    # Network Parameters
    "MODEL": {
        "OBSERVATION_DIM": 256,
        "RNN_DEPTH": 1,
        "RNN_HIDDEN_SIZE": 512,
        "RNN_DROPOUT": 0.2,
        # The value of sigma squared, corresponding to Eq. (11) in the paper. If the value is given, we will fix
        # to this value. If the value is None, we will estimate it from training data.
        "SIGMA2": None,
        # The value of p0, corresponding to Eq. (6) in the paper. If the value is given,
        # we will fix to this value. If the value is None, we will estimate it from training data using Eq. (13)
        # in the paper.
        "TRANSITION_BIAS": None,
        # The value of alpha for the Chinese restaurant process (CRP), corresponding to Eq. (7) in the paper.
        # In this open source implementation, currently we only support using a given value of crp_alpha.
        "CRP_ALPHA": 1.0,
        "VERBOSITY": 2,
    },
    "TRAIN": {
        "PATH": ROOT_DIR + "/models/uisrnn/training_data.npz",
        "SAVE": datetime.now().strftime(
            "./models/uisrnn/saved_model_uisrnn_%d-%m-%Y-%H-%M"
        ),
        # Whether to enforce cluster ID uniqueness across different training sequences. Only effective when the
        # first input to fit() is a list of sequences. In general, assume the cluster IDs for two sequences are
        # [a, b] and [a, c]. If the `a` from the two sequences are not the same label, then this arg should be True.
        "ENFORCE_CLUSTER_ID_UNIQUENESS": False,
        "BATCH_SIZE": 30,
        "LEARNING_RATE": 1e-4,
        "TRAIN_ITERATION": 3000,
        # The number of permutations per utterance sampled in the training data.'
        "NUM_PERMUTATIONS": 20,
        # Max norm of the gradient.
        "GRAD_MAX_NORM": 5.0,
        # The half life of the leaning rate for training. If this value is
        # positive, we reduce learning rate by half every this many
        # iterations during training. If this value is 0 or negative,
        # we do not decay learning rate.
        "LEARNING_RATE_HALF_LIFE": 1000,
        # The inverse gamma shape for estimating sigma2. This value is only meaningful when sigma2 is not given,
        # and estimated from data.
        "SIGMA_ALPHA": 1.0,
        # The inverse gamma scale for estimating sigma2. This value is only meaningful when sigma2 is not given,
        # and estimated from data.
        "SIGMA_BETA": 1.0,
        # The network regularization multiplicative.
        "REGULARIZATION_WEIGHT": 1e-5,
    },
    "INFERENCE": {
        # The beam search size for inference.
        "BEAM_SIZE": 10,
        # The number of look ahead steps during inference.
        "LOOK_AHEAD": 1,
        # During inference, we concatenate M duplicates of the test sequence, and run inference on this
        # concatenated sequence. Then we return the inference results on the last duplicate as the
        # final prediction for the test sequence.
        "TEST_ITERATION": 2,
    },
}
