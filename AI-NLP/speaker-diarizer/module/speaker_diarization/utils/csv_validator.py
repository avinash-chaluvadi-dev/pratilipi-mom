# This script is used to validate a particular file, w.r.t the ground truth csv.
# Keep the G.T. csv in the dataset folder and check the output csv from the test component with the G.T. folder.

import logging

import pandas as pd
from speaker_diarization import config


def validate(resp_csv):
    """
    params : the responce csv from the engine.
    returns : boolean flag, whether the two csvs match or not.
    """

    gt_df = pd.read_csv(config.GT_CSV)
    pred_df = pd.read_csv(resp_csv)
    logging.debug("Ground Truth Dataframe : ")
    logging.debug(gt_df)
    logging.debug("Prediction Dataframe : ")
    logging.debug(pred_df)

    if gt_df.equals(pred_df):
        return True
    else:
        return False
