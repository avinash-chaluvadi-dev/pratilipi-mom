import logging
import os
import time
from typing import Dict, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import DataLoader

from .. import config
from ..utils import utils_tools
from .data_loader import TransformerModelDataset
from .model import AllocatorBackbone, MLBackbone

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


def get_features(
    model: AllocatorBackbone,
    data: TransformerModelDataset,
    is_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to extract features from texts using given transformer model

    Parameters:
        model: Allocator Backbone model which will be used for the feature extraction
        data: Input data for the feature extraction process
        is_train: whether the mode is train or not

    Returns:
        Tuple consisting of extracted features and corresponding labels (if mode is train else None)

    """
    logging.debug("Extracting features for input data...")
    try:
        device = model.device
        model.to(device)
        labels = []
        features = []
        with torch.no_grad():
            for index, encodings in enumerate(data):
                # Loading data into GPU/ CPU
                batch_data = {key: data.to(device) for key, data in encodings.items()}
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)  # Adding extra dimension
                    attention_mask = attention_mask.unsqueeze(0)
                feature = model(input_ids=input_ids, attention_mask=attention_mask)
                features.extend(feature)
                if is_train is True:
                    label = encodings["labels"]
                    if label.dim() == 0:
                        label = label.unsqueeze(0)
                    labels.extend(label)
                del encodings  # Clearing encodings from memory
        logging.debug("Feature extraction completed...")
    except Exception:
        logging.exception("Error during feature extraction", exc_info=True)
        raise
    if is_train is True:
        return np.array(features), np.array(labels)
    else:
        return np.array(features), np.array([])


def kfold_cross_validation(
    transformer_model: AllocatorBackbone,
    data: TransformerModelDataset,
    is_train: bool = False,
    sequence_classification: bool = False,
    func_test: bool = False,
) -> Dict[str, float]:
    """
    Function to perform K fold Cross Validation (uses Stratified K fold strategy)

    Parameters:
        transformer_model: Allocator Backbone model for feature extraction or sequence classification
        data: Dataset for training / evaluation data to be used for training/ evaluation of the model
        is_train: whether or not the model mode is training (false if only evaluation to be done)
        sequence_classification: if the model provided is transformer model used for sequence classification (False means model used is
        transformer for feature extraction)
        func_test: True for model package testing


    Returns:
        Dictionary of list of f1 scores for k iterations of chosen algorithm

    """
    skfold = StratifiedKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )
    if sequence_classification is False:
        features, labels = get_features(transformer_model, data, is_train=is_train)
        classifier_model = MLBackbone(config.ML_CLF_MODEL)
    else:
        features, labels = data.texts, data.labels
        classifier_model = transformer_model

    if len(features) != len(labels):
        logging.exception("Length of features and labels are not equal.")
        raise IndexError("Features and labels are not of equal length")

    f1_scores = {}
    max_accuracy = -float("inf")
    scores = {}
    for train_index, test_index in skfold.split(features, labels):
        x_train, x_test = features[train_index], features[test_index]  # Training data
        y_train, y_test = labels[train_index], labels[test_index]  # Validation data
        if sequence_classification is False:
            if is_train is False:
                classifier_model = MLBackbone(config.ML_CLF_MODEL)
            try:
                scores = classifier_model.train(
                    train_data=(x_train, y_train), test_data=(x_test, y_test)
                )
            except Exception:
                logging.exception(f"Error Training {classifier_model.model_name}.")
                raise
            f1_score = scores["f1_score"]

            # Saving the ML model if the accuracy is greater than previous iteration
            if is_train is True and f1_score > max_accuracy:
                max_accuracy = f1_score
                if func_test is True:
                    model_name = config.ML_MODEL
                else:
                    model_name = config.BASE_ML_MODEL
                utils_tools.pickle_dump(classifier_model, model_name)
        else:
            if is_train is False:
                classifier_model = transformer_model

            # Initializing Transformer Model Dataset
            train_dataset = TransformerModelDataset(
                model=classifier_model,
                text_list=x_train,
                labels_list=y_train,
                is_train=is_train,
            )

            # Initializing Dataloader
            train_loader = DataLoader(
                train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
            )

            # training model
            scores = train_model(
                classifier_model,
                train_data=train_loader,
                eval_data=(x_test, y_test),
                save_model=False,
            )
            f1_score = scores["f1_score"]

            # Saving the model if the accuracy is greater than previous iteration
            if is_train is True and f1_score > max_accuracy:
                max_accuracy = f1_score
                if func_test is True:
                    model_name = config.FINETUNED_CLASSIFIER
                else:
                    model_name = config.BASE_FINETUNED_CLASSIFIER
                utils_tools.save_model(
                    model_name,
                    classifier_model.model,
                    classifier_model.tokenizer,
                )
        # Getting the f1_scores for every fold
        f1_scores[classifier_model.model_name] = f1_scores.get(
            classifier_model.model_name, []
        ) + [scores["f1_score"]]
    logging.info(f"F1 Scores: {f1_scores}")
    return scores


def train_model(
    model: AllocatorBackbone,
    train_data: Union[DataLoader, TransformerModelDataset],
    eval_data: Tuple[np.ndarray, np.ndarray] = None,
    sequence_classification: bool = True,
    save_model: bool = True,
    func_test: bool = False,
) -> Dict[str, float]:
    """
    Function to train the Sequence classification model

    Parameters:
        model: Transformer model for Sequence classification
        train_data: Data loader/ batcher for training data
        eval_data: Data loader/ batcher for evaluation data
        sequence_classification:if the model provided is transformer model used for sequence classification (False means model used is
        transformer for feature extraction)
        save_model: whether or not model is to be saved
        func_test: True for model package testing

    Returns:
        Dictionary containing evaluation scores (if eval_data is not None)

    """
    if sequence_classification is False:
        classifier_model = MLBackbone(config.ML_CLF_MODEL)
        features, labels = get_features(model, train_data)
        scores = classifier_model.train(
            train_data=(features, labels), test_data=eval_data
        )
        if func_test is True:
            model_name = config.ML_MODEL
        else:
            model_name = config.BASE_ML_MODEL

        # Dumping the ML model using pickle
        utils_tools.pickle_dump(classifier_model, model_name)
    else:
        start = time.time()
        device = model.device
        model.to(device)
        model.train()
        logging.debug(f"Training of {model.model_name} started...")
        try:
            for epoch in range(config.NUM_EPOCHS):
                logging.info(f"Epoch {epoch + 1} Started...")
                len_dataset = len(train_data)
                interval = max(len_dataset // 20, 1)
                for index, encodings in enumerate(train_data):
                    if (index + 1) % interval == 0:
                        logging.info(
                            f"{model.model_name} Epoch {epoch + 1} Iteration {index + 1} Running..."
                        )
                    model.optimizer.zero_grad()
                    # Loading Data into GPU/ CPU
                    batch_data = {
                        key: data.to(device) for key, data in encodings.items()
                    }
                    outputs = model(**batch_data)
                    loss = outputs[0]
                    loss.backward()
                    model.optimizer.step()
                    del batch_data  # Clearing batch_data from memory
            end = time.time()
            logging.debug(f"Training of {model.model_name} completed...")
            logging.info(
                f"[{model.model_name}] Training took {round(end - start, 2)} seconds"
            )
        except Exception:
            logging.exception(f"Error Training {model.model_name}")
            raise
        scores = {
            "Scores": "Only training of the model done. Provide evaluation dataset for the evaluation scores."
        }
        if eval_data is not None:
            x_test, y_test = eval_data
            scores = evaluate(model, x_test, y_test)

        # If train model is used individually save_model is True else save_model is False
        if save_model is True:
            if func_test is True:
                model_name = config.FINETUNED_CLASSIFIER
            else:
                model_name = config.BASE_FINETUNED_CLASSIFIER
            utils_tools.save_model(model_name, model.model, model.tokenizer)
        else:
            logging.warning(
                "Only training of the model done. Provide evaluation dataset for the evaluation scores."
            )
    return scores


def evaluate(model, x_test, y_test):
    """
    Wrapper function for model evaluation

    Parameters:
        model: Model used for evaluation
        x_test: Evaluation data
        y_test: Labels corresponding to evaluation data
    """

    if model.sequence_classification is False:
        dataset = TransformerModelDataset(
            model, text_list=x_test, labels_list=y_test, is_train=True
        )
        x_test, y_test = get_features(model, dataset, is_train=True)
        model = utils_tools.pickle_load(config.BASE_ML_MODEL)
    return model.evaluate(x_test, y_test)
