import os
import shutil
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm

from .. import config
from ..utils import custom_logging, utils_tools


class LabelClassifierEngine:
    """
    Label classification engine class: This class to encapsulate the train, serving and evaluating function of the
    Label classification
    """

    def __init__(
        self,
        model,
        data_loader,
        save_model=False,
        num_epochs=config.NUM_EPOCHS,
    ):
        self.model = model
        self.epochs = num_epochs
        self.save_model = save_model
        self.data_loader = data_loader
        self.logger = custom_logging.get_logger()
        self.optimizer = config.optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = nn.CrossEntropyLoss().to(self.device)

    def train_eval(self, fold):
        """
        Label classification training Method. This methods instantiates the data loader in the training mode and store
        the model after total training epochs have been completed.
        """
        if self.data_loader[1] is None:
            self.logger.exception(
                "Evaluation data_loader_object is missing, Change the mode to 'train'"
            )
            raise
        try:
            best_eval_accuracy = 0
            self.model.to(
                self.device
            )  # To move the model parameters to the available device.
            self.logger.debug("Starting the Training Loop ..")  # Training loop start
            for epoch in range(self.epochs):
                self.model.train()  # Call the train method from the nn.Module base class
                eval_loss = 0
                train_loss = 0
                eval_accuracy = 0
                train_accuracy = 0
                self.logger.debug(f"[INFO] Fold {fold+1} Epoch {epoch + 1} Started..")
                for index, batch in tqdm(enumerate(self.data_loader[0])):
                    self.logger.debug(
                        f"[INFO] [TRAINING] Fold {fold+1} Epoch {epoch + 1} Iteration {index + 1} Running.."
                    )
                    self.optimizer.zero_grad()
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    label = batch["label"].to(self.device)
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    loss = self.loss_function(output, label)
                    loss.backward()
                    self.optimizer.step()
                    train_loss = train_loss + loss.item()
                    _, hypothesis = torch.max(output, dim=1)
                    train_accuracy = (
                        train_accuracy
                        + torch.sum(torch.tensor(hypothesis == label)).item()
                    )
                self.logger.debug(
                    "Starting the Evaluation Loop .."
                )  # Evaluation loop start
                self.model.eval()  # Call the eval method from the nn.Module base class
                with torch.no_grad():
                    for index, batch in tqdm(enumerate(self.data_loader[1])):
                        self.logger.debug(
                            f"[INFO] [EVALUATION] Fold{fold + 1} Epoch {epoch + 1} Iteration {index + 1} Running.."
                        )
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        token_type_ids = batch["token_type_ids"].to(self.device)
                        label = batch["label"].to(self.device).to(self.device)
                        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        )
                        loss = self.loss_function(output, label)
                        eval_loss = eval_loss + loss.item()
                        _, hypothesis = torch.max(output, dim=1)
                        eval_accuracy = (
                            eval_accuracy
                            + torch.sum(torch.tensor(hypothesis == label)).item()
                        )
                train_accuracy = train_accuracy / (
                    len(self.data_loader[0])
                    * config.params.get("train").get("batch_size")
                )
                eval_accuracy = eval_accuracy / (
                    len(self.data_loader[1])
                    * config.params.get("eval").get("batch_size")
                )
                train_loss = train_loss / (
                    len(self.data_loader[0])
                    * config.params.get("train").get("batch_size")
                )
                eval_loss = eval_loss / (
                    len(self.data_loader[1])
                    * config.params.get("eval").get("batch_size")
                )
                if self.save_model and eval_accuracy >= best_eval_accuracy:
                    utils_tools.modify_specs(
                        specs={
                            "finetuned_model": config.LABEL_CLASSIFIER,
                            "best_eval_accuracy": best_eval_accuracy,
                        }
                    )
                    utils_tools.save_model(
                        logger=self.logger,
                        model_name=config.LABEL_CLASSIFIER,
                        model=self.model.model,
                        tokenizer=self.model.tokenizer,
                    )
                    best_eval_accuracy = eval_accuracy
                train_st = (
                    f"Training Loss: {train_loss} Train Accuracy: {train_accuracy}"
                )
                eval_st = (
                    f"Evaluation Loss: {eval_loss} Evaluation Accuracy: {eval_accuracy}"
                )
                self.logger.debug(
                    f"Fold {fold + 1} Epoch: {epoch} {train_st} {eval_st}"
                )
                self.logger.debug(f"[INFO] Fold {fold + 1} Epoch {epoch + 1} Ended..")
            return self.model, best_eval_accuracy
        except (RuntimeError, MemoryError, ValueError, TypeError):
            self.logger.exception("Training/Evaluation Exception Occurred")

    def train(self, num_epochs=None):
        """
        Training method for Label Classification. Saves the model after training of model is completed
        """
        accuracy = []
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
                for index, batch in tqdm(enumerate(self.data_loader[0])):
                    self.logger.debug(
                        f"[INFO] [TRAINING] Epoch {epoch + 1} Iteration {index + 1} Running.."
                    )
                    self.optimizer.zero_grad()
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    label = batch["label"].to(self.device).to(self.device)
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    loss = self.loss_function(output, label)
                    loss.backward()
                    self.optimizer.step()
                    train_loss = train_loss + loss.item()
                    _, hypothesis = torch.max(output, dim=1)
                    train_accuracy = (
                        train_accuracy
                        + torch.sum(torch.tensor(hypothesis == label)).item()
                    )
                train_accuracy = train_accuracy / (
                    len(self.data_loader[0])
                    * config.params.get("train").get("batch_size")
                )
                accuracy.append(train_accuracy)
                train_loss = train_loss / (
                    len(self.data_loader[0])
                    * config.params.get("train").get("batch_size")
                )
                train_st = (
                    f"Training Loss: {train_loss} Train Accuracy: {train_accuracy}"
                )
                self.logger.debug(f"Epoch: {epoch+1} {train_st}")
            self.logger.info("Model has been successfully built..")
            # utils_tools.save_model_bin(model_name=config.Label_CLASSIFIER, model=self.model)
            accuracy = sum(accuracy) / len(accuracy)
            utils_tools.modify_specs(specs={"train_accuracy": accuracy})
            best_train_accuracy = utils_tools.get_specs(
                logger=self.logger, key="train_accuracy"
            )
            if not best_train_accuracy:
                best_train_accuracy = 0
            if self.save_model and accuracy >= best_train_accuracy:
                utils_tools.modify_specs(specs={"train_accuracy": accuracy})
                model_path = utils_tools.get_specs(logger=self.logger, key="model")
                if model_path:
                    shutil.rmtree(os.path.join(config.MODELS_DIR, model_path))
                utils_tools.modify_specs(specs={"model": config.LABEL_CLASSIFIER})
                utils_tools.save_model(
                    logger=self.logger,
                    model_name=config.LABEL_CLASSIFIER,
                    model=self.model,
                    tokenizer=self.model.tokenizer,
                )
            elif self.save_model and accuracy < best_train_accuracy:
                finetuned_model_path = utils_tools.get_specs(
                    logger=self.logger, key="finetuned_model"
                )
                self.logger.info(
                    f"Train accuracy of {finetuned_model_path} is more than {config.LABEL_CLASSIFIER}"
                )
        except (RuntimeError, MemoryError, ValueError, TypeError):
            self.logger.exception("Training Exception Occurred")

    def eval(self):
        """
        Evaluation method for Label Classification. Saves the model after training of model is completed.
        """
        try:
            eval_loss = 0
            eval_accuracy = 0
            count = 0
            self.logger.debug(
                "Starting the Evaluation Loop .."
            )  # Evaluation loop start
            self.model.to(self.device)
            self.model.eval()  # Call the eval method from the nn.Module base class
            # l1
            with torch.no_grad():
                for index, batch in tqdm(enumerate(self.data_loader[0])):
                    count = count + batch["input_ids"].shape[0]
                    self.logger.debug(
                        f"[INFO] [EVALUATION] Iteration {index + 1} Running.."
                    )
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    label = batch["label"].to(self.device).to(self.device)
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    loss = self.loss_function(output, label)
                    eval_loss = eval_loss + loss.item()
                    _, hypothesis = torch.max(output, dim=1)
                    eval_accuracy = (
                        eval_accuracy
                        + torch.sum(torch.tensor(hypothesis == label)).item()
                    )
            eval_loss = eval_loss / (len(self.data_loader[0]))
            eval_accuracy = eval_accuracy / count
            eval_st = (
                f"Evaluation Loss: {eval_loss} Evaluation Accuracy: {eval_accuracy}"
            )
            self.logger.debug(f"{eval_st}")
            best_eval_accuracy = utils_tools.get_specs(
                logger=self.logger, key="best_eval_accuracy"
            )
            if best_eval_accuracy is None:
                utils_tools.modify_specs(specs={"best_eval_accuracy": eval_accuracy})
                if self.save_model:
                    utils_tools.save_model(
                        logger=self.logger,
                        model_name=config.LABEL_CLASSIFIER,
                        model=self.model.model,
                        tokenizer=self.model.tokenizer,
                    )
            if (
                eval_accuracy
                > utils_tools.get_specs(logger=self.logger, key="best_eval_accuracy")
                and self.save_model
            ):
                utils_tools.save_model(
                    logger=self.logger,
                    model_name=config.LABEL_CLASSIFIER,
                    model=self.model.model,
                    tokenizer=self.model.tokenizer,
                )
            return eval_loss, eval_accuracy

        except (RuntimeError, MemoryError, ValueError, TypeError):
            self.logger.exception("Evaluation Exception Occurred")

    def serve(self):
        """

        Serve method for Label Classification. Returns the predictions of input data.

        """
        try:
            text = []
            predictions = []
            confidence_score = []
            count = 0
            utils_tools.load_state_dict(logger=self.logger, model=self.model)
            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                for index, batch in tqdm(enumerate(self.data_loader[0])):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    prob_scores = f.softmax(output, dim=1)
                    for count in range(batch["input_ids"].shape[0]):
                        self.logger.debug(
                            f"[INFO] [INFERENCE] BATCH {index + 1} Sample {count + 1} Done.."
                        )
                    _, hypothesis = torch.max(output, dim=1)
                    count = count + 1
                    text.extend(batch["text"])
                    predictions.extend(
                        [config.CLASSIFICATION_LABELS[index] for index in hypothesis]
                    )
                    confidence_score.extend(
                        torch.max(prob_scores, dim=1)[0].detach().cpu().tolist()
                    )
                    # speaker_ids.extend(batch["speaker_id"].detach().cpu().tolist())
                    # chunk_ids.extend(batch["chunk_id"].detach().cpu().tolist())
            confidence_score = [score * 100 for score in confidence_score]
            output_dict = {
                "text": text,
                "predictions": predictions,
                "confidence_score": confidence_score,
            }
            response = {
                "label_classification_result": {
                    "status": "Success",
                    "details": output_dict,
                }
            }
            return response
        except RuntimeError:
            self.logger.exception(
                "Exception encountered while serving the Label Classifier Engine",
                exc_info=True,
            )
            response = {
                "label_classification_result": {
                    "status": "Error",
                    "details": f"{traceback.format_exc()}",
                }
            }
            return response
