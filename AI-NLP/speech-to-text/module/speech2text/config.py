import os
from datetime import datetime

import tensorflow as tf

from module.speech2text.seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from module.speech2text.seq2seq.decoders import FullyConnectedCTCDecoder
from module.speech2text.seq2seq.models import Speech2Text
from module.speech2text.seq2seq.encoders import TDNNEncoder
from module.speech2text.seq2seq.losses import CTCLoss
from module.speech2text.seq2seq.optimizers.lr_policies import poly_decay
from module.speech2text.seq2seq.optimizers.novograd import NovoGrad

residual_dense = True  # Enable or disable Dense Residual

base_model = Speech2Text
ROOT_DIR = "/var/s3fs-demofs/Pratilipi/AI/speech-to-text/"
LOAD_DIR = os.path.join(ROOT_DIR, "dataset")
OUTPUT_RUN = os.path.join(ROOT_DIR, "results", "test_outputs")
OUTPUT_LOG = os.path.join(ROOT_DIR, "results", "run_logs")
OUT_JSON = datetime.now().strftime("output_%H-%M-%d-%m-%Y.json")
IMAGE_CLASSIFIER = datetime.now().strftime("FineTunedClassifier_%H-%M-%d-%m-%Y.pth")

base_params = {
    "random_seed": 0,
    "use_horovod": False,
    "num_epochs": 400,
    "num_gpus": 1,
    "batch_size_per_gpu": 1,
    "iter_size": 1,
    "save_summaries_steps": 100,
    "print_loss_steps": 10,
    "print_samples_steps": 2200,
    "eval_steps": 2200,
    "save_checkpoint_steps": 1100,
    "logdir": "/var/s3fs-demofs/Pratilipi/AI/speech-to-text/models/jasper_model/",
    "num_checkpoints": 2,
    "optimizer": NovoGrad,
    "optimizer_params": {
        "beta1": 0.95,
        "beta2": 0.98,
        "epsilon": 1e-08,
        "weight_decay": 0.001,
        "grad_averaging": False,
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 0.02,
        "min_lr": 1e-5,
        "power": 2.0,
    },
    "larc_params": {
        "larc_eta": 0.001,
    },
    "dtype": "mixed",
    "loss_scaling": "Backoff",
    "summaries": [
        "learning_rate",
        "variables",
        "gradients",
        "larc_summaries",
        "variable_norm",
        "gradient_norm",
        "global_gradient_norm",
    ],
    "encoder": TDNNEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d",
                "repeat": 1,
                "kernel_size": [11],
                "stride": [2],
                "num_channels": 256,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [11],
                "stride": [1],
                "num_channels": 256,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.8,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [11],
                "stride": [1],
                "num_channels": 256,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.8,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [13],
                "stride": [1],
                "num_channels": 384,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.8,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [13],
                "stride": [1],
                "num_channels": 384,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.8,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [17],
                "stride": [1],
                "num_channels": 512,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.8,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [17],
                "stride": [1],
                "num_channels": 512,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.8,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [21],
                "stride": [1],
                "num_channels": 640,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.7,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [21],
                "stride": [1],
                "num_channels": 640,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.7,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [25],
                "stride": [1],
                "num_channels": 768,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.7,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 5,
                "kernel_size": [25],
                "stride": [1],
                "num_channels": 768,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.7,
                "residual": True,
                "residual_dense": residual_dense,
            },
            {
                "type": "conv1d",
                "repeat": 1,
                "kernel_size": [29],
                "stride": [1],
                "num_channels": 896,
                "padding": "SAME",
                "dilation": [2],
                "dropout_keep_prob": 0.6,
            },
            {
                "type": "conv1d",
                "repeat": 1,
                "kernel_size": [1],
                "stride": [1],
                "num_channels": 1024,
                "padding": "SAME",
                "dilation": [1],
                "dropout_keep_prob": 0.6,
            },
        ],
        "dropout_keep_prob": 0.7,
        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            "uniform": False,
        },
        "normalization": "batch_norm",
        "activation_fn": tf.nn.relu,
        "data_format": "channels_last",
        "use_conv_mask": True,
    },
    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "initializer": tf.contrib.layers.xavier_initializer,
        "use_language_model": False,
        # params for decoding the sequence with language model
        # "beam_width": 2048,
        # "alpha": 2.0,
        # "beta": 1.5,
        # "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
        # "lm_path": "language_model/4-gram.binary",
        # "trie_path": "language_model/trie.binary",
        # "alphabet_config_path": "seq2seq/test_utils/toy_speech_data/vocab.txt",
        "infer_logits_to_pickle": False,
    },
    "loss": CTCLoss,
    "loss_params": {},
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "module/speech2text/seq2seq/test_utils/toy_speech_data/vocab.txt",
        "norm_per_feature": True,
        "window": "hanning",
        "precompute_mel_basis": True,
        "sample_freq": 16000,
        "pad_to": 16,
        "dither": 1e-5,
        "backend": "librosa",
    },
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "augmentation": {
            "speed_perturbation_ratio": [0.9, 1.0, 1.1],
        },
        "dataset_type": "csv",
        "dataset_files": [
            "dataset/train.csv"
        ],
        "max_duration": 16.7,
        "shuffle": True,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "dataset_type": "json",
        "dataset_files": [
            "dataset/eval.json",
        ],
        "shuffle": False,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "dataset_type": "json",
        "dataset_files": [
            "dataset/diarization_output.json",
        ],
        "shuffle": False,
    },
}
json_data_params = {
    "diarization_output_top_level_key": "speaker_diarization",
    "audio_path": "path",
    "speaker_id": "speaker_id"
}
backend = "librosa"
