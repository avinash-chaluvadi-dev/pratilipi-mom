import json
import os

import pandas as pd
from torch.utils.data import DataLoader

from .. import config
from ..utils import custom_logging, utils_tools
from . import data_loader
from .cross_validation import CrossValidation
from .engine import LabelClassifierEngine
from .model import LabelBackbone


def network(
    mode,
    save_model,
    checkpoint,
    tokenizer,
    fold=None,
    func_test=None,
    dataframe=None,
    json_data=None,
):
    """

    Function to initialise the Label engine for using the model in train, eval or serve
        :param mode: mode of Label classifier model (train, eval or serve)
        :param save_model: Whether to save the model after training.
        :param checkpoint: Name of the checkpoint used to restore the model
        :param tokenizer: tokenizer from huggingface library
        :param fold: Fold Number
        :param func_test: True for functional testing of package
        :param dataframe: Input dataframe for the model
        :param json_data: Input JSON for the model
        :returns: Label Engine class which contains the
                  method train, evaluate and serve for the
                  Label Classifier model

    """
    if mode in ["train_eval"] or (
        mode == "package_test" and func_test in ["train_eval"]
    ):
        train_df = dataframe.loc[dataframe["kf"] != fold, :]
        eval_df = dataframe.loc[dataframe["kf"] == fold, :]
        train_dataset = data_loader.LabelDataset.from_dataframe(
            dataframe=train_df, mode=mode, tokenizer=tokenizer
        )
        eval_dataset = data_loader.LabelDataset.from_dataframe(
            dataframe=eval_df, mode=mode, tokenizer=tokenizer
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=config.params.get("train").get("batch_size"),
            shuffle=True,
        )

        eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=config.params.get("eval").get("batch_size"),
            shuffle=True,
        )
        # Instantiate Model Creator Class, load the model architecture and the weights.
        backbone_model = LabelBackbone(model=checkpoint, tokenizer=tokenizer)
        model_engine = LabelClassifierEngine(
            model=backbone_model,
            save_model=save_model,
            data_loader=(train_data_loader, eval_data_loader),
        )
        return model_engine
    elif mode in ["train"] or (mode == "package_test" and func_test in ["train"]):
        train_dataset = data_loader.LabelDataset.from_dataframe(
            dataframe=dataframe, mode=mode, tokenizer=tokenizer, func_test=func_test
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=config.params.get("train").get("batch_size"),
            shuffle=True,
        )
        backbone_model = LabelBackbone(model=checkpoint, tokenizer=tokenizer)
        model_engine = LabelClassifierEngine(
            model=backbone_model,
            save_model=save_model,
            data_loader=(train_data_loader, None),
        )
        return model_engine
    elif mode in ["eval"] or (mode == "package_test" and func_test in ["eval"]):
        eval_dataset = data_loader.LabelDataset.from_dataframe(
            dataframe=dataframe, mode=mode, tokenizer=tokenizer, func_test=func_test
        )
        eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=config.params.get("eval").get("batch_size"),
            shuffle=True,
        )
        backbone_model = LabelBackbone(model=checkpoint, tokenizer=tokenizer)
        model_engine = LabelClassifierEngine(
            model=backbone_model,
            save_model=save_model,
            data_loader=(eval_data_loader, None),
        )
        return model_engine
    elif mode in ["serve"] or (mode == "package_test" and func_test in ["serve"]):
        if json_data is None:
            serve_dataset = data_loader.LabelDataset.from_dataframe(
                dataframe=dataframe, mode=mode, tokenizer=tokenizer, func_test=func_test
            )
            serve_data_loader = DataLoader(
                serve_dataset,
                batch_size=config.params.get("serve").get("batch_size"),
                shuffle=True,
            )
            backbone_model = LabelBackbone(model=checkpoint, tokenizer=tokenizer)
            model_engine = LabelClassifierEngine(
                model=backbone_model,
                save_model=save_model,
                data_loader=(serve_data_loader, None),
            )
            return model_engine
        elif dataframe is None:
            serve_dataset = data_loader.LabelDataset.from_json(
                json_data=json_data, mode=mode, tokenizer=tokenizer, func_test=func_test
            )
            serve_data_loader = DataLoader(
                serve_dataset,
                batch_size=config.params.get("train").get("batch_size"),
                shuffle=True,
            )
            backbone_model = LabelBackbone(model=checkpoint, tokenizer=tokenizer)
            model_engine = LabelClassifierEngine(
                model=backbone_model,
                save_model=save_model,
                data_loader=(serve_data_loader, None),
            )
            return model_engine


def run(**kwargs):
    """
    Pluggable engine for Label Classification package.
    """
    mode = kwargs.get("mode")
    save_model = kwargs.get("save_model")
    checkpoint = kwargs.get("checkpoint")
    tokenizer = kwargs.get("tokenizer")
    logger = custom_logging.get_logger()
    best_accuracy = 0
    accuracy = []
    if mode == "train_eval":
        if config.params.get("train").get("dataset_type") == "csv":
            dataframes = []
            for csv_file in config.params.get("train").get("dataset_files"):
                dataframe_path = os.path.join(config.LOAD_DIR, csv_file)
                dataframes.append(pd.read_csv(dataframe_path))
            dataframe = pd.concat(dataframes, ignore_index=True)
            cv = CrossValidation(
                dataframe=dataframe,
                target_columns=config.TARGET_COLUMNS,
                num_folds=config.NUM_FOLDS,
            )
            dataframe = cv.split()
            logger.info("Instance of LabelBackbone is created")
            logger.info("Instance of LabelClassifierEngine is created")
            for fold in range(config.NUM_FOLDS):
                logger.debug(f"[INFO] Fold {fold + 1} started")
                model_engine = network(
                    mode=mode,
                    save_model=save_model,
                    checkpoint=checkpoint,
                    tokenizer=tokenizer,
                    fold=fold,
                    dataframe=dataframe,
                )
                model, eval_accuracy = model_engine.train_eval(fold=fold)
                accuracy.append(eval_accuracy)
                if eval_accuracy >= best_accuracy:
                    best_accuracy = eval_accuracy
                    utils_tools.save_model(
                        logger=logger, model_name=config.Label_CLASSIFIER, model=model
                    )

        elif config.params.get("train").get("dataset_type") == "json":
            dataframes = []
            for json_file in config.params.get("eval").get("dataset_files"):
                json_path = os.path.join(config.LOAD_DIR, json_file)
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                    dataframes.append(utils_tools.create_dataframe(json_data=json_data))
            dataframe = pd.concat(dataframes, ignore_index=True)
            cv = CrossValidation(
                dataframe=dataframe,
                target_columns=config.TARGET_COLUMNS,
                num_folds=config.NUM_FOLDS,
            )
            dataframe = cv.split()
            logger.info("Instance of LabelBackbone is created")
            logger.info("Instance of LabelClassifierEngine is created")
            for fold in range(config.NUM_FOLDS):
                logger.debug(f"[INFO] Fold {fold + 1} started")
                model_engine = network(
                    mode=mode,
                    save_model=save_model,
                    checkpoint=checkpoint,
                    tokenizer=tokenizer,
                    fold=fold,
                    dataframe=dataframe,
                )
                model, eval_accuracy = model_engine.train_eval(fold=fold)
                accuracy.append(eval_accuracy)
                if eval_accuracy >= best_accuracy:
                    best_accuracy = eval_accuracy
                    utils_tools.save_model(
                        logger=logger, model_name=config.Label_CLASSIFIER, model=model
                    )

    elif mode == "train":
        if config.params.get("train").get("dataset_type") == "csv":
            dataframes = []
            for csv_file in config.params.get("train").get("dataset_files"):
                dataframe_path = os.path.join(config.LOAD_DIR, csv_file)
                dataframes.append(pd.read_csv(dataframe_path))
            dataframe = pd.concat(dataframes, ignore_index=True)
            logger.info("Instance of LabelBackbone is created")
            logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=mode,
                save_model=save_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                dataframe=dataframe,
            )
            model_engine.train()
        elif config.params.get("train").get("dataset_type") == "json":
            dataframes = []
            for json_file in config.params.get("train").get("dataset_files"):
                json_path = os.path.join(config.LOAD_DIR, json_file)
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                    dataframes.append(utils_tools.create_dataframe(json_data=json_data))
            dataframe = pd.concat(dataframes, ignore_index=True)
            logger.info("Instance of LabelBackbone is created")
            logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=mode,
                save_model=save_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                dataframe=dataframe,
            )
            model_engine.train()

    elif mode == "eval":
        if config.params.get("eval").get("dataset_type") == "csv":
            dataframes = []
            for csv_file in config.params.get("eval").get("dataset_files"):
                dataframe_path = os.path.join(config.LOAD_DIR, csv_file)
                dataframes.append(pd.read_csv(dataframe_path))
            dataframe = pd.concat(dataframes, ignore_index=True)
            logger.info("Instance of LabelBackbone is created")
            logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=mode,
                save_model=save_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                dataframe=dataframe,
            )
            _, evaluation_accuracy = model_engine.eval()
            logger.debug(f"Evaluation Accuracy: {evaluation_accuracy}")
        elif config.params.get("eval").get("dataset_type") == "json":
            dataframes = []
            for json_file in config.params.get("eval").get("dataset_files"):
                json_path = os.path.join(config.LOAD_DIR, json_file)
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                    dataframes.append(utils_tools.create_dataframe(json_data=json_data))
            dataframe = pd.concat(dataframes, ignore_index=True)
            logger.info("Instance of LabelBackbone is created")
            logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=mode,
                save_model=save_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                dataframe=dataframe,
            )
            _, evaluation_accuracy = model_engine.eval()
            logger.debug(f"Evaluation Accuracy: {evaluation_accuracy}")

    elif mode == "serve":
        if config.params.get("serve").get("dataset_type") == "csv":
            dataframes = []
            for csv_file in config.params.get("serve").get("dataset_files"):
                dataframe_path = os.path.join(config.LOAD_DIR, csv_file)
                dataframes.append(pd.read_csv(dataframe_path))
            dataframe = pd.concat(dataframes, ignore_index=True)
            logger.info("Instance of LabelBackbone is created")
            logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=mode,
                save_model=save_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                dataframe=dataframe,
            )
            model_engine.serve()
        elif config.params.get("serve").get("dataset_type") == "json":
            json_data = []
            for json_file in config.params.get("serve").get("dataset_files"):
                json_path = os.path.join(config.LOAD_DIR, json_file)
                with open(json_path, "r") as f:
                    json_data.append(json.load(f))
            logger.info("Instance of LabelBackbone is created")
            logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=mode,
                save_model=save_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                json_data=json_data,
            )
            response = model_engine.serve()
            logger.info(response)
            if response.get("Label_classification_result").get("status") == "Success":
                utils_tools.save_result(json_data=response)
                return response
            else:
                logger.info(response)
                return response


class PackageTest:
    """
    Package test for functional testing of the Label Classification package.
    """

    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode")
        self.tokenizer = kwargs.get("tokenizer")
        self.func_test = kwargs.get("func_test")
        self.logger = custom_logging.get_logger()
        self.checkpoint = kwargs.get("checkpoint")
        self.save_model = kwargs.get("save_model")

    def test_train_component(self):
        if config.params.get("package_test").get("dataset_type") == "csv":
            dataframes = []
            for csv_file in config.params.get("package_test").get("dataset_files"):
                dataframe_path = os.path.join(config.LOAD_DIR, csv_file)
                dataframes.append(pd.read_csv(dataframe_path))
            dataframe = pd.concat(dataframes, ignore_index=True)
            self.logger.info("Instance of LabelBackbone is created")
            self.logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=self.mode,
                save_model=self.save_model,
                checkpoint=self.checkpoint,
                tokenizer=self.tokenizer,
                func_test=self.func_test,
                dataframe=dataframe,
            )
            model_engine.train()
        elif config.params.get("package_test").get("dataset_type") == "json":
            dataframes = []
            for json_file in config.params.get("package_test").get("dataset_files"):
                json_path = os.path.join(config.LOAD_DIR, json_file)
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                    dataframes.append(utils_tools.create_dataframe(json_data=json_data))
            dataframe = pd.concat(dataframes, ignore_index=True)
            self.logger.info("Instance of LabelBackbone is created")
            self.logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=self.mode,
                save_model=self.save_model,
                checkpoint=self.checkpoint,
                tokenizer=self.tokenizer,
                func_test=self.func_test,
                dataframe=dataframe,
            )
            model_engine.train()

    def test_eval_component(self):
        if config.params.get("package_test").get("dataset_type") == "csv":
            dataframes = []
            for csv_file in config.params.get("package_test").get("dataset_files"):
                dataframe_path = os.path.join(config.LOAD_DIR, csv_file)
                dataframes.append(pd.read_csv(dataframe_path))
            dataframe = pd.concat(dataframes, ignore_index=True)
            self.logger.info("Instance of LabelBackbone is created")
            self.logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=self.mode,
                save_model=self.save_model,
                checkpoint=self.checkpoint,
                tokenizer=self.tokenizer,
                func_test=self.func_test,
                dataframe=dataframe,
            )
            model_engine.eval()
        elif config.params.get("package_test").get("dataset_type") == "json":
            dataframes = []
            for json_file in config.params.get("package_test").get("dataset_files"):
                json_path = os.path.join(config.LOAD_DIR, json_file)
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                    dataframes.append(utils_tools.create_dataframe(json_data=json_data))
            dataframe = pd.concat(dataframes, ignore_index=True)
            self.logger.info("Instance of LabelBackbone is created")
            self.logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=self.mode,
                save_model=self.save_model,
                checkpoint=self.checkpoint,
                tokenizer=self.tokenizer,
                func_test=self.func_test,
                dataframe=dataframe,
            )
            model_engine.eval()

    def test_serve_component(self):
        if config.params.get("serve").get("dataset_type") == "csv":
            dataframes = []
            for csv_file in config.params.get("serve").get("dataset_files"):
                dataframe_path = os.path.join(config.LOAD_DIR, csv_file)
                dataframes.append(pd.read_csv(dataframe_path))
            dataframe = pd.concat(dataframes, ignore_index=True)
            self.logger.info("Instance of LabelBackbone is created")
            self.logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=self.mode,
                save_model=self.save_model,
                checkpoint=self.checkpoint,
                tokenizer=self.tokenizer,
                func_test=self.func_test,
                dataframe=dataframe,
            )
            self.logger.debug(model_engine.serve())

        elif config.params.get("serve").get("dataset_type") == "json":
            json_data = []
            for json_file in config.params.get("serve").get("dataset_files"):
                json_path = os.path.join(config.LOAD_DIR, json_file)
                with open(json_path, "r") as f:
                    json_data.append(json.load(f))
            self.logger.info("Instance of LabelBackbone is created")
            self.logger.info("Instance of LabelClassifierEngine is created")
            model_engine = network(
                mode=self.mode,
                save_model=self.save_model,
                checkpoint=self.checkpoint,
                tokenizer=self.tokenizer,
                func_test=self.func_test,
                json_data=json_data,
            )
            self.logger.debug(model_engine.serve())
