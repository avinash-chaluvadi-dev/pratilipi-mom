import argparse
import json
import os

import numpy as np
from torch.utils.data import DataLoader

from . import config
from .ml_engine import data_loader
from .ml_engine.cross_validation import CrossValidation
from .ml_engine.engine import FrameClassifierEngine
from .ml_engine.model import FrameClassifier
from .utils import custom_logging, utils_tools


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--extract_frames",
        help="can be True or False to extract frames",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--mode",
        help="train, eval, and serve mode to run the sentiment engine",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save_model",
        help="can be True or False to save the model",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--func_test",
        help="run the package in the testing mode, value = all, runs all the test."
        "Value = train/serve/eval, runs the package evaluation in a specific mode",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--framework",
        help="pytorch or keras to create the model network",
        type=str,
        required=False,
    )
    return parser.parse_args()


def run(**kwargs):
    """
    # :param mode: Instantiate the Sentiment engine using a particular mode, train, eval or serve.
    # :param extract_frames: Boolean value to extract_frames.
    # :return: return the Sentiment engine object
    """
    mode = kwargs.get("mode")
    extract_frames = kwargs.get("extract_frames")
    save_model = kwargs.get("save_model")
    logger = custom_logging.get_logger()

    if mode == "train_eval":
        # Instantiate the dataset object, parsing through the dataset folder.
        # This reads the dataset from the DatasetCreator class.
        if config.USE_S3:
            raise NotImplementedError(
                "Keyframe Classifier does not support model training using S3 currently."
            )
        dataset_creator = data_loader.DatasetCreator(
            video_dir=config.VIDEO_DIR, frames_dir=config.FRAMES_DIR
        )
        if extract_frames:
            dataset_creator.video_to_frames()
        dataframe = dataset_creator.create_train_data()
        cv = CrossValidation(
            dataframe=dataframe,
            target_columns=config.TARGET_COLUMNS,
            num_folds=config.NUM_FOLDS,
        )
        dataframe = cv.split()
        best_accuracy = 0
        accuracy = []
        logger.info("Instance of FrameClassifier is created")
        logger.info("Instance of FrameClassifierEngine is created")
        for fold in range(config.NUM_FOLDS):
            logger.debug(f"[INFO] Fold {fold+1} started")
            train_df = dataframe.loc[dataframe["kf"] != fold, :]
            eval_df = dataframe.loc[dataframe["kf"] == fold, :]
            train_dataset = data_loader.ImageDataset.from_dataframe(
                dataframe=train_df, mode=mode
            )
            eval_dataset = data_loader.ImageDataset.from_dataframe(
                dataframe=eval_df, mode=mode
            )
            train_data_loader = DataLoader(
                train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True
            )
            eval_data_loader = DataLoader(
                eval_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=True
            )
            # Instantiate Model Creator Class, load the model architecture and the weights.
            backbone_model = FrameClassifier()
            model_engine = FrameClassifierEngine(
                model=backbone_model,
                save_model=save_model,
                train_data_loader=train_data_loader,
                eval_data_loader=eval_data_loader,
            )
            model, eval_accuracy = model_engine.train_eval(fold=fold)
            accuracy.append(eval_accuracy)
            if eval_accuracy >= best_accuracy:
                best_accuracy = eval_accuracy
                utils_tools.save_model(model_name=config.IMAGE_CLASSIFIER, model=model)
        logger.debug(f"Mean Evaluation Accuracy: {np.array(accuracy).mean()}")

        #  TODO save accuracy and best model_path in configuration file

    elif mode == "train":
        # This reads the dataframe from the DatasetCreator class.
        if config.USE_S3:
            raise NotImplementedError(
                "Keyframe Classifier does not support model training using S3 currently."
            )
        dataset_creator = data_loader.DatasetCreator(
            video_dir=config.VIDEO_DIR, frames_dir=config.FRAMES_DIR
        )
        if extract_frames:
            dataset_creator.video_to_frames()
        dataframe = dataset_creator.create_train_data()
        # Instantiate the dataset object, parsing through the dataset folder.
        train_dataset = data_loader.ImageDataset.from_dataframe(
            dataframe=dataframe, mode=mode
        )
        # Instantiate the Data-loader.
        train_data_loader = DataLoader(
            train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True
        )
        logger.info("Instance of DataLoader class is created")
        # Instantiate Model Creator Class, load the model architecture and the weights.
        backbone_model = FrameClassifier()
        logger.info("Instance of FrameClassifier is created")
        model_engine = FrameClassifierEngine(
            model=backbone_model,
            save_model=save_model,
            train_data_loader=train_data_loader,
        )
        logger.info("Instance of FrameClassifierEngine is created")
        model_engine.train()

    elif mode == "eval":
        evaluation_accuracy = {}
        eval_params = config.params["eval"]
        if eval_params["dataset_type"] in ["json"]:
            for json_file in eval_params["dataset_files"]:
                if config.USE_S3:
                    json_data = {}
                else:
                    with open(os.path.join(config.LOAD_DIR, json_file)) as f:
                        json_data = json.load(f)

                dataset = data_loader.ImageDataset.from_json(
                    json_data=json_data, mode=mode
                )
                # Instantiate the Data-loader.
                eval_data_loader = DataLoader(
                    dataset,
                    batch_size=config.params.get("infer").get("batch_size"),
                )
                # Instantiate Model Creator Class, load the model architecture and the weights.
                backbone_model = FrameClassifier()
                logger.info("Instance of FrameClassifier is created")
                model_engine = FrameClassifierEngine(
                    model=backbone_model,
                    save_model=save_model,
                    eval_data_loader=eval_data_loader,
                )
                logger.info("Instance of FrameClassifierEngine is created")
                _, eval_accuracy = model_engine.eval()
                evaluation_accuracy[json_file] = eval_accuracy
            logger.debug(f"Evaluation Accuracy: {evaluation_accuracy}")

        elif eval_params["dataset_type"] in ["csv"]:
            for json_file in eval_params["dataset_files"]:
                if config.USE_S3:
                    json_data = {}
                else:
                    with open(os.path.join(config.LOAD_DIR, json_file)) as f:
                        json_data = json.load(f)

                dataset = data_loader.ImageDataset.from_json(
                    json_data=json_data, mode=mode
                )
                # Instantiate the Data-loader.
                eval_data_loader = DataLoader(
                    dataset,
                    batch_size=config.params.get("infer").get("batch_size"),
                )
                # Instantiate Model Creator Class, load the model architecture and the weights.
                backbone_model = FrameClassifier()
                logger.info("Instance of FrameClassifier is created")
                model_engine = FrameClassifierEngine(
                    model=backbone_model,
                    save_model=save_model,
                    eval_data_loader=eval_data_loader,
                )
                logger.info("Instance of FrameClassifierEngine is created")
                _, eval_accuracy = model_engine.eval()
                evaluation_accuracy[json_file] = eval_accuracy
            logger.debug(f"Evaluation Accuracy: {evaluation_accuracy}")

    elif mode == "serve":
        final_response = {}
        serve_params = config.params["infer"]
        if serve_params["dataset_type"] in ["json"]:
            for json_file in serve_params["dataset_files"]:
                if config.USE_S3:
                    json_data = {}
                else:
                    with open(os.path.join(config.LOAD_DIR, json_file)) as f:
                        json_data = json.load(f)
                dataset = data_loader.ImageDataset.from_json(
                    json_data=json_data, mode=mode
                )
                serve_data_loader = DataLoader(
                    dataset,
                    batch_size=config.params.get("infer").get("batch_size"),
                )
                backbone_model = FrameClassifier()
                model_engine = FrameClassifierEngine(
                    model=backbone_model,
                    serve_data_loader=serve_data_loader,
                )
                response = model_engine.serve()
                keyframe_classifier_output = utils_tools.create_output_json(
                    json_data=json_data,
                    response=response.get("keyframe_classification_result").get(
                        "details"
                    ),
                )
                final_response[json_file] = keyframe_classifier_output
        logger.info(f"The response from the Model : {final_response}")

    else:
        logger.exception("Invalid mode for network creation..")


class PackageTest:
    def __init__(self, **kwargs):
        self.logger = custom_logging.get_logger()
        self.save_model = kwargs.get("save_model")
        self.mode = kwargs.get("mode")
        self.func_test = kwargs.get("func_test")

    def test_train_component(self):
        train_params = config.params["train"]
        if "json" in train_params["dataset_type"]:
            for json_file in train_params["package_test_files"]:
                with open(os.path.join(config.LOAD_DIR, json_file)) as f:
                    json_data = json.load(f)
                    dataset = data_loader.ImageDataset.from_json(
                        json_data=json_data,
                        mode=self.mode,
                        func_test=self.func_test,
                    )
                    # Instantiate the Data-loader.
                    train_data_loader = DataLoader(
                        dataset,
                        batch_size=config.params.get("train").get("batch_size"),
                    )
                    # Instantiate Model Creator Class, load the model architecture and the weights.
                    backbone_model = FrameClassifier()
                    self.logger.info("Instance of FrameClassifier is created")
                    model_engine = FrameClassifierEngine(
                        model=backbone_model,
                        save_model=self.save_model,
                        train_data_loader=train_data_loader,
                    )
                    self.logger.info("Instance of FrameClassifierEngine is created")
                    model_engine.train(
                        num_epochs=config.params.get("train").get("package_test_epochs")
                    )
        else:
            self.logger.exception("Invalid file type for testing serve component")
            raise

    def test_eval_component(self):
        eval_params = config.params["eval"]
        if "json" in eval_params["dataset_type"]:
            for json_file in eval_params["package_test_files"]:
                with open(os.path.join(config.LOAD_DIR, json_file)) as f:
                    json_data = json.load(f)
                    dataset = data_loader.ImageDataset.from_json(
                        json_data=json_data,
                        mode=self.mode,
                        func_test=self.func_test,
                    )
                    # Instantiate the Data-loader.
                    eval_data_loader = DataLoader(
                        dataset,
                        batch_size=config.params.get("eval").get("batch_size"),
                    )
                    # Instantiate Model Creator Class, load the model architecture and the weights.
                    backbone_model = FrameClassifier()
                    self.logger.info("Instance of FrameClassifier is created")
                    model_engine = FrameClassifierEngine(
                        model=backbone_model,
                        save_model=self.save_model,
                        eval_data_loader=eval_data_loader,
                    )
                    self.logger.info("Instance of FrameClassifierEngine is created")
                    model_engine.eval()
        else:
            self.logger.exception("Invalid file type for testing serve component")
            raise

    def test_serve_component(self):
        final_response = {}
        serve_params = config.params["infer"]
        if "json" in serve_params["dataset_type"]:
            for json_file in serve_params["package_test_files"]:
                with open(os.path.join(config.LOAD_DIR, json_file)) as f:
                    json_data = json.load(f)
                    dataset = data_loader.ImageDataset.from_json(
                        json_data=json_data,
                        mode=self.mode,
                        func_test=self.func_test,
                    )
                    # Instantiate the Data-loader.
                    serve_data_loader = DataLoader(
                        dataset,
                        batch_size=config.params.get("infer").get("batch_size"),
                    )
                    # Instantiate Model Creator Class, load the model architecture and the weights.
                    backbone_model = FrameClassifier()
                    self.logger.info("Instance of FrameClassifier is created")
                    model_engine = FrameClassifierEngine(
                        model=backbone_model,
                        save_model=self.save_model,
                        serve_data_loader=serve_data_loader,
                    )
                    self.logger.info("Instance of FrameClassifierEngine is created")
                    response = model_engine.serve()
                    keyframe_classifier_output = utils_tools.create_output_json(
                        json_data=json_data,
                        response=response.get("keyframe_classification_result").get(
                            "details"
                        ),
                    )
                    final_response[json_file] = keyframe_classifier_output
            utils_tools.save_result(json_data=final_response)
            self.logger.debug(f"The response from the Model : {final_response}")
        else:
            self.logger.exception("Invalid file type for testing serve component")
            raise


def main():
    logger = custom_logging.get_logger()
    cmd_args = arg_parser()
    if cmd_args.mode == "package_test":
        package_test_obj = PackageTest(
            save_model=cmd_args.save_model,
            mode=cmd_args.mode,
            func_test=cmd_args.func_test,
        )
        if cmd_args.func_test == "all":
            logger.debug("Testing all the components of the package..")
            logger.debug("Testing Training Component..")
            package_test_obj.test_train_component()
            logger.debug("Training Component Test Passed..")
            logger.debug("Testing Evaluation Component..")
            package_test_obj.test_eval_component()
            logger.debug("Evaluation Component Test Passed..")
            logger.debug("Testing Serving Component..")
            package_test_obj.test_serve_component()
            logger.debug("Serving Component Test Passed..")
            # package_test(test_mode=cmd_args.func_test)
        elif cmd_args.func_test == "train":
            logger.debug("Testing Training Component..")
            package_test_obj.test_train_component()
            logger.debug("Training Component Test Passed..")
        elif cmd_args.func_test == "eval":
            logger.debug("Testing Evaluation Component..")
            package_test_obj.test_eval_component()
            logger.debug("Evaluation Component Test Passed..")
        elif cmd_args.func_test == "serve":
            logger.debug("Testing Serving Component..")
            package_test_obj.test_serve_component()
            logger.debug("Serving Component Test Passed..")
        else:
            logger.exception("Invalid functional test mode")

    else:
        if cmd_args.mode == "train_eval":
            logger.debug(
                "Instantiating model Frame Classifier Engine for training and evaluation.."
            )
            run(
                mode=cmd_args.mode,
                extract_frames=cmd_args.extract_frames,
                save_model=cmd_args.save_model,
            )

        elif cmd_args.mode == "train":
            logger.debug("Instantiating model Frame Classifier Engine for training..")
            run(
                mode=cmd_args.mode,
                extract_frames=cmd_args.extract_frames,
                save_model=cmd_args.save_model,
            )

        elif cmd_args.mode == "eval":
            logger.debug("Instantiating model Frame Classifier Engine for evaluation..")
            run(
                mode=cmd_args.mode,
                extract_frames=cmd_args.extract_frames,
                save_model=cmd_args.save_model,
            )

        elif cmd_args.mode == "serve":
            logger.debug("Instantiating model Frame Classifier Engine for Serving..")
            run(
                mode=cmd_args.mode,
                extract_frames=cmd_args.extract_frames,
                save_model=cmd_args.save_model,
            )

        else:
            logger.exception("Invalid mode Argument..", exc_info=True)


if __name__ == "__main__":
    main()
