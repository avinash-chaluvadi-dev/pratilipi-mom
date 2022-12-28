"""
Highest level python script for the Allocator (Deadline Escalation Classifier). This script can be used to run the package in the
standalone mode which enables the Allocator (Deadline and Escalation classifier) to be trained, evaluated, served and tested agnostic of the
application interface code.

"""
import argparse
import logging
import os
from typing import Dict, Optional

import config
from torch.utils.data.dataloader import DataLoader

from .ml_engine.data_loader import TransformerModelDataset
from .ml_engine.engine import AllocatorEngine
from .ml_engine.model import AllocatorBackbone
from .utils import utils_tools

# Adding the StreamHandler to print the logfile output to the stderr stream.
logging.getLogger().addHandler(logging.StreamHandler())
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


def arg_parser():
    """
    ArgumentParser for parsing of the various arguments passed

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="train, eval, and serve mode to run the sentiment engine",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--func_test",
        help="run the package in the testing mode, value = all, runs all the test."
        "Value = train/serve/eval, runs the package evaluation in a specific mode",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--file_path",
        help="Input data file path for serving the model",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    return args


def network(mode: str, json_data: Optional[Dict] = None, func_test: bool = False):
    """
    Function to initialise the Allocator engine for using the model in train, eval or serve

    Parameters:
        mode: mode of Allocator (Deadline and Escalation classifier) model (train, eval or serve)
        json_data: input JSON for the model
        func_test: True for functional testing of package

    Returns:
        Allocator Engine class which contains the method train, evaluate and serve for the Allocator(Deadline and Escalation classifier) Classifier model

    """
    # Initialize Model Creator Class, load the model architecture and the weights.
    model_name = config.TRANSFORMER_MODEL
    if mode == "train":
        is_train = True
        shuffle = True
    else:
        is_train = False
        shuffle = False
        if config.SEQUENCE_CLASSIFICATION is True:
            if func_test is True:
                model_name = config.FINETUNED_CLASSIFIER
            else:
                model_name = config.BASE_FINETUNED_CLASSIFIER

    backbone_model = AllocatorBackbone(
        model_name=model_name,
        sequence_classification=config.SEQUENCE_CLASSIFICATION,
    )

    # Initializing the dataset
    dataset = TransformerModelDataset(
        backbone_model, json_data=json_data, is_train=is_train
    )

    # Initialize the Data-loader.
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)

    # Run the Sentiment Engine.
    model_engine = AllocatorEngine(
        model=backbone_model,
        data_loader=data_loader,
        dataset=dataset,
        sequence_classification=config.SEQUENCE_CLASSIFICATION,
    )
    return model_engine


def train(func_test: bool = False):
    """
    Wrapper function for Training of the Allocator model engine

    Parameters:
        func_test: True for model's package testing

    """
    model_engine = network("train", func_test=func_test)
    model_engine.train(kfold_training=config.KFOLD_TRAINING, func_test=func_test)


def evaluate(func_test: bool = False):
    """
    Wrapper function for Evaluation of the Allocator model engine

    """
    model_engine = network("evaluate")
    model_engine.evaluate(kfold_validation=config.KFOLD_VALIDATION)


def serve(json_data: Dict = None, save_result: bool = False):
    """
    Wrapper function for serving of the Allocator model engine

    Parameters:
        json_data: Input JSON for serving
        save_result: whether to save result locally or not (True for functional testing)

    Returns:
        API response for the input JSON

    """
    model_engine = network("serve", json_data=json_data)
    response = model_engine.serve(save_result=save_result)
    return response


def package_test(test_mode: str):
    """
    Package test for functional testing of the Allocator Classifier package

    Parameters:
        test_mode: mode of the functional testing(all, train, evaluate or serve)

    """
    if test_mode == "all":
        logging.debug("Testing all the components of the package..")
        logging.debug("Testing Training Component..")
        train(func_test=True)
        logging.debug("Training Component test passed..")

        # logging.debug("Testing Evaluation Component..")
        # evaluate()
        # logging.debug("Evaluation Component test passed..")

        logging.debug("Testing Serving Component")
        serve(save_result=True)
        logging.debug("Serving component test passed..")

    elif test_mode == "train":
        logging.debug("Testing Training component..")
        train(func_test=True)
        logging.debug("Training component test passed..")

    elif test_mode == "serve":
        logging.debug("Testing Serving component")
        serve(save_result=True)
        logging.debug("Serving component test passed..")

    else:
        logging.debug("Testing Evaluation component")
        evaluate()
        logging.debug("Evaluating Component passed..")


def main():
    """
    Main function for the Allocator Classifier package

    """
    cmd_args = arg_parser()
    if cmd_args.func_test:

        # Running the functional test
        logging.debug(
            "Initializing Allocator (Deadline and Escalation classifier) package's functional testing.."
        )
        package_test(cmd_args.func_test)

    else:
        if cmd_args.mode == "train":
            #  Training the model
            logging.debug(
                "Initializing Allocator (Deadline and Escalation classifier) model for training.."
            )
            model_engine = network(cmd_args.mode)
            model_engine.train(kfold_training=config.KFOLD_TRAINING)

        elif cmd_args.mode == "eval":
            # Evaluating the model
            logging.debug(
                "Initializing Allocator (Deadline and Escalation classifier) model for Evaluation.."
            )
            logging.debug("Evaluation component NOT Supported at the present..")
            # model_engine = network(cmd_args.mode)
            # model_engine.evaluate()

        elif cmd_args.mode == "serve":
            # Serving process of the model
            logging.debug(
                "Initializing Allocator (Deadline and Escalation classifier) model for Serving.."
            )
            if cmd_args.file_path:
                json_data = utils_tools.load_json("", cmd_args.file_path)
            else:
                json_data = None
            model_engine = network(cmd_args.mode, json_data)
            response = model_engine.serve()
            logging.debug(f"The response from the Model : {response}")

        elif cmd_args.mode == "package_test":

            if not cmd_args.func_test:
                logging.exception(
                    " Pass an argument for func_test to run the package for testing.."
                )
            else:
                logging.debug(
                    "Instantiating Sentiment Engine package's functional testing.."
                )
                package_test(cmd_args.func_test)

        else:
            logging.exception("Invalid Argument mode argument passed..", exc_info=True)


if __name__ == "__main__":
    main()
