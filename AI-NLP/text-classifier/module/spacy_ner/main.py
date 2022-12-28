import argparse
import json
import os

import pandas as pd

from . import config
from .ml_engine import data_loader, engine, model
from .utils import custom_logging, utils_tools


def arg_parser():
    parser = argparse.ArgumentParser()

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
    return parser.parse_args()


def run(**kwargs):
    """
    # :param mode: Instantiate the NER engine using a particular mode, train, eval or serve.
    """
    mode = kwargs.get("mode")
    save_model = kwargs.get("save_model")
    logger = custom_logging.get_logger()

    if mode == "train":
        train_params = config.params["train"]
        # Instantiate the dataset object, parsing through the dataset folder.
        # This reads the dataset from the DatasetCreator class.
        dataset_creator = data_loader.DatasetCreator(
            train_file=train_params["dataset_files"][0]
        )
        dataframe = dataset_creator.create_train_data()

        train_data_loader = data_loader.NERDataset.from_dataframe(
            dataframe=dataframe, mode=mode, params=train_params
        )

        # Instantiate Model Creator Class, load the model architecture and the weights.
        backbone_model = model.SpacyNER.from_spacy_model(config.MODEL_NAME_LOAD)
        logger.info("Instance of SpacyNER is created")

        model_engine = engine.NEREngine(
            model=backbone_model,
            save_model=save_model,
            train_data_loader=train_data_loader.data,
        )
        logger.info("Instance of NEREngine is created")

        model_engine.train()
        logger.info(f"Training of NER model completed")

    elif mode == "eval":
        evaluation_accuracy = {}
        eval_params = config.params["eval"]
        for file in eval_params["dataset_files"]:
            if ".csv" in file:
                dataframe = pd.read_csv(os.path.join(config.LOAD_DIR, file))
                eval_data_loader = data_loader.NERDataset.from_dataframe(
                    dataframe=dataframe, mode=mode, params=eval_params
                )
            elif ".json" in file:
                with open(os.path.join(config.LOAD_DIR, file)) as f:
                    json_data = json.load(f)
                    eval_data_loader = data_loader.NERDataset.from_json(
                        json_data=json_data, mode=mode, params=eval_params
                    )
            else:
                logger.exception("Invalid file type for eval component")
                continue

            # Instantiate Model Creator Class, load the model.
            backbone_model = model.SpacyNER.from_spacy_model(config.BEST_MODEL)
            logger.info("Instance of SpacyNER is created")

            model_engine = engine.NEREngine(
                model=backbone_model,
                save_model=save_model,
                eval_data_loader=eval_data_loader.data,
            )
            logger.info("Instance of NEREngine is created")

            scores = model_engine.eval()
            evaluation_accuracy[file] = scores
        logger.info(f"Evaluation Scores of each file: {evaluation_accuracy}")

    elif mode == "serve":
        final_predictions = {}
        serve_params = config.params["serve"]
        for file in serve_params["dataset_files"]:
            if ".json" in file:
                with open(os.path.join(config.LOAD_DIR, file)) as f:
                    json_data = json.load(f)
                serve_data_loader = data_loader.NERDataset.from_json(
                    json_data=json_data, mode=mode, params=serve_params
                )

                backbone_model = model.SpacyNER.from_spacy_model(config.BEST_MODEL)
                logger.info("Instance of SpacyNER is created")

                model_engine = engine.NEREngine(
                    model=backbone_model,
                    save_model=save_model,
                    serve_data_loader=serve_data_loader.data,
                )
                logger.info("Instance of NEREngine is created")

                predictions = model_engine.serve()
                final_predictions[file] = predictions
            else:
                logger.exception("Invalid file type for serve component")
        logger.debug(f"Final response from the Model : {final_predictions}")

    else:
        logger.exception("Invalid mode for network creation..")


class PackageTest:
    def __init__(self, **kwargs):
        self.logger = custom_logging.get_logger()
        self.save_model = kwargs.get("save_model")
        self.mode = kwargs.get("mode")

    def test_train_component(self):
        """
        Method to test the training functionality of the NER Engine.
        """
        # Instantiate Model Creator Class, load the model
        backbone_model = model.SpacyNER.from_spacy_model(config.MODEL_NAME_LOAD)
        self.logger.info("Instance of SpacyNER is created")

        params = config.params["train"]
        for file in params["package_test_files"]:
            if ".json" in file:
                with open(os.path.join(config.LOAD_DIR, file)) as f:
                    json_data = json.load(f)
                    train_data_loader = data_loader.NERDataset.from_json(
                        json_data=json_data, mode="eval", params=params
                    )
            elif ".csv" in file:
                dataframe = pd.read_csv(os.path.join(config.LOAD_DIR, file))
                train_data_loader = data_loader.NERDataset.from_dataframe(
                    dataframe=dataframe, mode="train", params=params
                )
            else:
                self.logger.exception("Invalid file type for testing serve component")
                continue

            model_engine = engine.NEREngine(
                model=backbone_model,
                save_model=self.save_model,
                train_data_loader=train_data_loader.data,
            )
            self.logger.info("Instance of NEREngine is created")
            model_engine.train(num_epochs=params["package_test_epochs"])
            self.logger.info(f"Trained NER model with {file}")

    def test_eval_component(self):
        backbone_model = model.SpacyNER.from_spacy_model(config.BEST_MODEL)
        self.logger.info("Instance of SpacyNER is created")

        evaluation_accuracy = {}
        params = config.params["eval"]
        for file in params["package_test_files"]:
            if ".json" in file:
                with open(os.path.join(config.LOAD_DIR, file)) as f:
                    json_data = json.load(f)
                    eval_data_loader = data_loader.NERDataset.from_json(
                        json_data=json_data, mode="eval", params=params
                    )
            elif ".csv" in file:
                dataframe = pd.read_csv(os.path.join(config.LOAD_DIR, file))
                eval_data_loader = data_loader.NERDataset.from_dataframe(
                    dataframe=dataframe, mode="eval", params=params
                )
            else:
                self.logger.exception("Invalid file type for testing eval component")
                continue

            model_engine = engine.NEREngine(
                model=backbone_model, eval_data_loader=eval_data_loader.data
            )
            self.logger.info("Instance of NEREngine is created")

            scores = model_engine.eval()
            evaluation_accuracy[file] = scores
        self.logger.info(f"Evaluation Scores of each file: {evaluation_accuracy}")

    def test_serve_component(self):
        """
        Method to test the serving functionality of the Sentiment Engine. Using the function_test.json file.
        """
        final_predictions = {}
        backbone_model = model.SpacyNER.from_spacy_model(config.BEST_MODEL)
        self.logger.info("Instance of SpacyNER is created")

        params = config.params["serve"]
        for file in params["package_test_files"]:
            if ".json" in file:
                with open(os.path.join(config.LOAD_DIR, file)) as f:
                    json_data = json.load(f)
                serve_data_loader = data_loader.NERDataset.from_json(
                    json_data=json_data, mode="serve", params=params
                )
            else:
                self.logger.exception("Invalid file type for testing serve component")
                continue

            model_engine = engine.NEREngine(
                model=backbone_model, serve_data_loader=serve_data_loader.data
            )
            self.logger.info("Instance of NEREngine is created")

            response = model_engine.serve()
            final_predictions[file] = response
            self.logger.debug(f"Predictions for {file}: {response}")
        utils_tools.save_result(json_data=final_predictions)
        self.logger.debug(f"Final response from the Model : {final_predictions}")


def main():
    logger = custom_logging.get_logger()
    cmd_args = arg_parser()
    package_test_obj = PackageTest(save_model=cmd_args.save_model, mode="train")
    if cmd_args.mode == "package_test":
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
        if cmd_args.mode == "train":
            logger.debug("Instantiating model NER Engine for training..")
            run(
                mode=cmd_args.mode,
                save_model=cmd_args.save_model,
            )

        elif cmd_args.mode == "eval":
            logger.debug("Instantiating model NER Engine for evaluation..")
            run(
                mode=cmd_args.mode,
                save_model=cmd_args.save_model,
            )

        elif cmd_args.mode == "serve":
            logger.debug("Instantiating model NER Engine for Serving..")
            run(
                mode=cmd_args.mode,
                save_model=cmd_args.save_model,
            )

        else:
            logger.exception("Invalid mode Argument..", exc_info=True)


if __name__ == "__main__":
    main()
