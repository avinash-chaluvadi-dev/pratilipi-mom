import logging
import os
import time
from typing import Dict, List, Union

from rouge_score import rouge_scorer
from torch.utils.data.dataloader import DataLoader

from .. import config
from ..utils import utils_tools
from .model import HeadlinerBackbone

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


def train(model: HeadlinerBackbone, train_data: DataLoader) -> HeadlinerBackbone:
    """
    Function to train the T5 Headline model

    Parameters:
        model: Headliner backbone (T5 model)
        train_data: Data loader/ batcher for training data

    Returns:
        Fine tuned T5 Headline model

    """
    t0 = time.time()
    device = model.device
    model.to(device)
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        logging.info(f"[{model.model_name}] Epoch {epoch + 1} Started..")
        len_dataset = len(train_data)
        interval = max(len_dataset // 20, 1)
        for index, batch in enumerate(train_data):
            if (index + 1) % interval == 0:
                logging.info(
                    f"[{model.model_name}] Epoch {epoch + 1} Iteration {index + 1} Running.."
                )
            model.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)  # Loading Data into GPU/ CPU
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            loss = outputs[0]
            loss.backward()
            model.optimizer.step()
            del batch  # Clearing batch_data from memory

    t1 = time.time()
    logging.debug(f"[{model.model_name}] Training took {round(t1 - t0, 2)} seconds")
    return model


def train_model(
    model: HeadlinerBackbone, train_data: Union[DataLoader, List], func_test: bool
) -> None:
    """
    Wrapper function for training of Headliner model

    Parameters:
        model: Headliner Backbone (T5 model) for training
        train_data: Input dataset for training
        func_test: True for functional testing of package

    """
    finetuned_model = train(model, train_data)

    # Saving the test model if functional testing is True else Base model
    if func_test is True:
        model_name = config.FINETUNED_MODEL
    else:
        model_name = config.BASE_FINETUNED_MODEL
    utils_tools.save_model(model_name, finetuned_model.model, finetuned_model.tokenizer)


def evaluate(
    model: HeadlinerBackbone,
    x_test: Union[DataLoader, List],
    y_test: Union[DataLoader, List],
) -> Dict[str, float]:
    """
    Wrapper function for model evaluation

    Parameters:
        model: Model used for evaluation
        x_test: Evaluation data
        y_test: summaries corresponding to evaluation data

    Returns:
        Dictionary of evaluation scores

    """
    machine_headline = model.predict(x_test)  # Getting the prediction
    gold_headline = y_test.copy()
    scores = _evaluate_headline(gold_headline, machine_headline)
    return scores


def _evaluate_headline(gold_headline: List, machine_headline: List) -> Dict[str, float]:
    """
    Getting Rogue Scores for predicted headline and actual headline

    Parameters:
        gold_headline: list of actual summaries
        machine_headline: list of predicted summaries

    Returns:
        Dictionary of evaluation scores


    """
    if len(gold_headline) != len(machine_headline) or len(gold_headline) == 0:
        logging.exception(
            "Length of machine headline is not equal to actual headline..."
        )
        raise IndexError("Length of machine headline is not equal to actual headline.")

    precision_list = []
    recall_list = []
    fmeasure_list = []

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"], use_stemmer=True
    )  # scorer class for calculating rogue scores

    for h1, h2 in zip(gold_headline, machine_headline):
        scores = scorer.score(h1, h2)
        precision_list.append(scores["rouge1"].precision)
        recall_list.append(scores["rouge1"].recall)
        fmeasure_list.append(scores["rouge1"].fmeasure)

    # Finding Average scores for precision, Recall and f measure
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_fmeasure = sum(fmeasure_list) / len(fmeasure_list)
    scores = {
        "Average Precision": avg_precision,
        "Average Recall": avg_recall,
        "Average F Measure": avg_fmeasure,
    }
    logging.info(f"Scores: {scores}")
    return scores
