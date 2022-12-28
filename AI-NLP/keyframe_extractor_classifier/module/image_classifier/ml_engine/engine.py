import os
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm

from .. import config
from ..utils import custom_logging, utils_tools


class FrameClassifierEngine:
    """
    Image classifier engine class: This class to encapsulate the train, serving and evaluating function of the
    Image classifier
    """

    def __init__(
        self,
        model,
        save_model=False,
        train_data_loader=None,
        eval_data_loader=None,
        serve_data_loader=None,
    ):
        self.model = model
        self.save_model = save_model
        self.logger = custom_logging.get_logger()
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.serve_data_loader = serve_data_loader
        self.optimizer = config.optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = nn.CrossEntropyLoss().to(self.device)

    def train_eval(self, fold, num_epochs=None):
        """
        Image classifier training Method. This methods instantiates the data loader in the training mode and store
        the model after total training runs have been completed.
        """
        if num_epochs is None:
            epochs = config.NUM_EPOCHS
        else:
            epochs = num_epochs
        try:
            best_eval_accuracy = 0
            self.model.to(
                self.device
            )  # To move the model parameters to the available device.
            self.logger.debug("Starting the Training Loop ..")  # Training loop start
            for epoch in range(epochs):
                self.model.train()  # Call the train method from the nn.Module base class
                eval_loss = 0
                train_loss = 0
                eval_accuracy = 0
                train_accuracy = 0
                self.logger.debug(f"[INFO] Fold {fold+1} Epoch {epoch + 1} Started..")
                for index, batch in tqdm(enumerate(self.train_data_loader)):
                    self.logger.debug(
                        f"[INFO] [TRAINING] Fold {fold+1} Epoch {epoch + 1} Iteration {index + 1} Running.."
                    )
                    self.optimizer.zero_grad()
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    outputs = self.model(inputs=images)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss = train_loss + loss.item()
                    _, hypothesis = torch.max(outputs, dim=1)
                    train_accuracy = (
                        train_accuracy
                        + torch.sum(torch.tensor(hypothesis == labels)).item()
                    )

                self.logger.debug(
                    "Starting the Evaluation Loop .."
                )  # Evaluation loop start
                self.model.eval()  # Call the eval method from the nn.Module base class
                with torch.no_grad():
                    for index, batch in tqdm(enumerate(self.eval_data_loader)):
                        self.logger.debug(
                            f"[INFO] [EVALUATION] Fold{fold+1} Epoch {epoch + 1} Iteration {index + 1} Running.."
                        )
                        images = batch["image"].to(self.device)
                        labels = batch["label"].to(self.device)
                        outputs = self.model(inputs=images)
                        loss = self.loss_function(outputs, labels)
                        eval_loss = eval_loss + loss.item()
                        _, hypothesis = torch.max(outputs, dim=1)
                        eval_accuracy = (
                            eval_accuracy
                            + torch.sum(torch.tensor(hypothesis == labels)).item()
                        )
                train_accuracy = train_accuracy / (
                    len(self.train_data_loader) * config.TRAIN_BATCH_SIZE
                )
                eval_accuracy = eval_accuracy / (
                    len(self.eval_data_loader) * config.EVAL_BATCH_SIZE
                )
                train_loss = train_loss / (
                    len(self.train_data_loader) * config.TRAIN_BATCH_SIZE
                )
                eval_loss = eval_loss / (
                    len(self.eval_data_loader) * config.EVAL_BATCH_SIZE
                )
                if self.save_model > eval_accuracy >= best_eval_accuracy:
                    utils_tools.save_model(
                        model_name=config.IMAGE_CLASSIFIER, model=self.model
                    )
                    best_eval_accuracy = eval_accuracy
                train_st = (
                    f"Training Loss: {train_loss} Train Accuracy: {train_accuracy}"
                )
                eval_st = (
                    f"Evaluation Loss: {eval_loss} Evaluation Accuracy: {eval_accuracy}"
                )
                self.logger.debug(f"Fold {fold+1} Epoch: {epoch} {train_st} {eval_st}")
                self.logger.debug(f"[INFO] Fold {fold+1} Epoch {epoch + 1} Ended..")
            return self.model, best_eval_accuracy
        except (RuntimeError, MemoryError, ValueError, TypeError):
            self.logger.exception("Training/Evaluation Exception Occurred")

    def train(self, num_epochs=None):
        if num_epochs is None:
            epochs = config.NUM_EPOCHS
        else:
            epochs = num_epochs
        try:
            self.model.to(
                self.device
            )  # To move the model parameters to the available device.
            self.model.train()  # Call the train method from the nn.Module base class
            self.logger.debug("Starting the Training Loop ..")  # Training loop start
            for epoch in range(epochs):
                train_loss = 0
                train_accuracy = 0
                self.logger.debug(f"[INFO] Epoch {epoch + 1} Started..")
                for index, batch in tqdm(enumerate(self.train_data_loader)):
                    self.logger.debug(
                        f"[INFO] [TRAINING] Epoch {epoch + 1} Iteration {index + 1} Running.."
                    )
                    self.optimizer.zero_grad()
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    outputs = self.model(inputs=images)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss = train_loss + loss.item()
                    _, hypothesis = torch.max(outputs, dim=1)
                    train_accuracy = (
                        train_accuracy
                        + torch.sum(torch.tensor(hypothesis == labels)).item()
                    )
                train_accuracy = train_accuracy / (
                    len(self.train_data_loader) * config.TRAIN_BATCH_SIZE
                )
                train_loss = train_loss / (
                    len(self.train_data_loader) * config.TRAIN_BATCH_SIZE
                )
                train_st = (
                    f"Training Loss: {train_loss} Train Accuracy: {train_accuracy}"
                )
                self.logger.debug(f"Epoch: {epoch} {train_st}")
            if self.save_model:
                utils_tools.save_model(
                    model_name=config.IMAGE_CLASSIFIER, model=self.model
                )
        except (RuntimeError, MemoryError, ValueError, TypeError):
            self.logger.exception("Training Exception Occurred")

    def eval(self):
        try:
            eval_loss = 0
            eval_accuracy = 0
            count = 0
            self.logger.debug(
                "Starting the Evaluation Loop .."
            )  # Evaluation loop start
            self.model.load_state_dict(
                torch.load(
                    os.path.join(config.MODELS_DIR, config.params["train"]["model"])
                )
            )
            self.model.to(self.device)
            self.model.eval()  # Call the eval method from the nn.Module base class
            with torch.no_grad():
                for index, batch in tqdm(enumerate(self.eval_data_loader)):
                    count = count + batch["image"].shape[0]
                    self.logger.debug(
                        f"[INFO] [EVALUATION] Iteration {index + 1} Running.."
                    )
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    outputs = self.model(inputs=images)
                    loss = self.loss_function(outputs, labels)
                    eval_loss = eval_loss + loss.item()
                    _, hypothesis = torch.max(outputs, dim=1)
                    eval_accuracy = (
                        eval_accuracy
                        + torch.sum(torch.tensor(hypothesis == labels)).item()
                    )
            eval_loss = eval_loss / (len(self.eval_data_loader))
            eval_accuracy = eval_accuracy / count
            eval_st = (
                f"Evaluation Loss: {eval_loss} Evaluation Accuracy: {eval_accuracy}"
            )
            self.logger.debug(f"{eval_st}")
            if (
                eval_accuracy > config.params.get("eval").get("accuracy")
                and self.save_model
            ):
                utils_tools.save_model(
                    model_name=config.IMAGE_CLASSIFIER, model=self.model
                )
            return eval_loss, eval_accuracy

        except (RuntimeError, MemoryError, ValueError, TypeError):
            self.logger.exception("Evaluation Exception Occurred")

    def serve(self):
        try:
            predictions = []
            speaker_ids = []
            chunk_ids = []
            confidence_score = []
            count = 0
            if torch.cuda.is_available() and config.DEVICE == "cuda":
                self.model.load_state_dict(config.model)
            else:
                self.model.load_state_dict(config.model)
            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                for index, batch in tqdm(enumerate(self.serve_data_loader)):
                    images = batch["image"].to(self.device)
                    outputs = self.model(inputs=images)
                    prob_scores = f.softmax(outputs, dim=1)
                    for count in range(batch["image"].shape[0]):
                        self.logger.debug(
                            f"[INFO] [INFERENCE] BATCH {index+1} Sample {count + 1} Done.."
                        )
                    _, hypothesis = torch.max(outputs, dim=1)
                    count = count + 1
                    predictions.extend(
                        [config.CLASSIFICATION_LABELS[index] for index in hypothesis]
                    )
                    confidence_score.extend(
                        torch.max(prob_scores, dim=1)[0].detach().cpu().tolist()
                    )
                    speaker_ids.extend(batch["speaker_id"])
                    chunk_ids.extend(batch["chunk_id"].detach().cpu().tolist())
            confidence_score = [score * 100 for score in confidence_score]
            output_dict = {
                "predictions": predictions,
                "confidence_score": confidence_score,
                "speaker_ids": speaker_ids,
                "chunk_ids": chunk_ids,
            }
            response = {
                "keyframe_classification_result": {
                    "status": "Success",
                    "details": output_dict,
                }
            }
            return response
        except Exception:
            self.logger.exception(
                "Exception encountered while serving the KeyFrame Classifier Engine",
                exc_info=True,
            )
            response = {
                "keyframe_classification_result": {
                    "status": "Error",
                    "details": f"{traceback.format_exc()}",
                }
            }
            return response
