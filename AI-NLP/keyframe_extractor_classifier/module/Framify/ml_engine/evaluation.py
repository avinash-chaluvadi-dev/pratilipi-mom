import logging
import os
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .. import config
from ..utils import utils_tools

if not config.USE_EFS:
    logging.basicConfig(
        filename=os.path.join(config.OUTPUT_LOG, config.LOG_FILE),
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )


def get_closest_match(
    image: np.ndarray, ground_truth_images: List, threshold: float
) -> int:
    """
    Function to get the closest match of an image from the ground truth. Returns None if no image matches

    Parameters:
        image: numpy array loaded image to be used for comparisons.
        ground_truth_images: List of ground truth images to be compared with.
        threshold: Threshold value used for comparison by SSIM (Structural similarity Index Measure)

    Returns:
        index of the image from ground truth images found closest to the given image

    """
    closest_idx = None
    closest_diff = -float("inf")
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
    for idx, img in enumerate(ground_truth_images):
        gray_ground_truth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        difference = ssim(
            gray_img, gray_ground_truth
        )  # Calculating the SSIM value for both the images
        if difference >= max(threshold, closest_diff):
            # if the SSIM value is more than the maximum of threshold or previous closest SSIM value
            closest_idx = idx
            closest_diff = difference
    return closest_idx


def get_metrics(y_true: List, y_pred: List) -> Dict[str, float]:
    """
    Function used for getting different evaluation metrics

    Parameters:
        y_true: ground truth values dictionary values
        y_pred: Predicted frames dictionary values

    Returns:
        Dictionary of evaluation scores

    """
    false_positive = y_pred.count(False)  # Calculating False Positives
    false_negative = y_true.count(False)  # Calculating False Negatives
    true_positive = y_true.count(True)  # Calculating True Positives
    precision = utils_tools.precision_score(
        true_positive, false_positive
    )  # calculating precision
    recall = utils_tools.recall_score(
        true_positive, false_negative
    )  # Calculating Recall
    f1 = utils_tools.f1_score(
        true_positive, false_positive, false_negative
    )  # Calculating F1 Scores
    scores = {"precision_score": precision, "recall_score": recall, "f1_score": f1}
    return scores


def get_scores(
    prediction_dir: Union[str, Path], ground_truth_dir: Union[str, Path]
) -> Dict[str, float]:
    """
    Function used for evaluation of the Framify model

    Parameters:
        prediction_dir: Directory path containing the predicted frames
        ground_truth_dir: Directory path containing the ground truth frames

    Returns:
        Dictionary of evaluation scores

    """
    # Checking if provided paths exists or not
    if not os.path.exists(prediction_dir):
        logging.exception(f"{prediction_dir} does not exist...")
        raise FileNotFoundError(f"{prediction_dir} does not exist...")

    if not os.path.exists(ground_truth_dir):
        logging.exception(f"{ground_truth_dir} does not exist...")
        raise FileNotFoundError(f"{ground_truth_dir} does not exist...")

    # Reading the images from directory
    ground_truth_file = [
        cv2.imread(os.path.join(ground_truth_dir, file))
        for file in os.listdir(ground_truth_dir)
        if file.endswith(f".{config.FRAME_FORMAT}")
    ]
    prediction_file = [
        cv2.imread(os.path.join(prediction_dir, file))
        for file in os.listdir(prediction_dir)
        if file.endswith(f".{config.FRAME_FORMAT}")
    ]

    # Resizing the images
    ground_truth_images = [cv2.resize(file, config.dsize) for file in ground_truth_file]
    prediction_images = [cv2.resize(file, config.dsize) for file in prediction_file]

    # Creating dictionary for marking the images
    # (will be used in calculating True positive, False Positive and False Negative)
    ground_truth_dict = {ind: False for ind in range(len(ground_truth_images))}
    prediction_dict = {ind: False for ind in range(len(prediction_images))}

    # Iterating over every predicted frames and finding it's closest match from the ground truth frames
    for idx, image in enumerate(prediction_images):
        index = get_closest_match(image, ground_truth_images, threshold=0.7)
        if index is not None:
            # if a frame was found as a match
            ground_truth_dict[index] = True
            prediction_dict[idx] = True
    y_true = list(ground_truth_dict.values())
    y_pred = list(prediction_dict.values())
    return get_metrics(y_true, y_pred)
