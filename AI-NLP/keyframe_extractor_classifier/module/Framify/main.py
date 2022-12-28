"""
Highest level python script for the Framify (Keyframe Extractor). This script can be used to run the package in the
standalone mode which enables the Keyframe extractor to be evaluated, served and tested agnostic of the
application interface code.

"""
import argparse
import logging
import os
import sys
from typing import Dict, Optional

import boto3

from . import config
from .ml_engine.data_loader import FramifyDataset
from .ml_engine.engine import FramifyEngine
from .ml_engine.model import FramifyBackbone
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


def network(json_data: Optional[Dict] = None) -> FramifyEngine:
    """
    Function to initialise the Framify engine for using the model in eval or serve mode

    Parameters:
        json_data: input JSON for the model

    Returns:
        Framify Engine class which contains the method evaluate and serve for the Framify model

    """
    backbone_model = FramifyBackbone()  # Loading model
    dataset = FramifyDataset(json_data=json_data)  # Initialising dataset
    model_engine = FramifyEngine(backbone_model, dataset)  # Initialising Framify Engine
    return model_engine


def evaluate():
    """
    Wrapper function for Evaluation of the Summarizer model engine
    """
    logging.info("Evaluation Component to be set up...")
    model_engine = network()
    if config.EVAL_VIDEO is None:
        input_file = config.DEFAULT_EVAL_VIDEO
        ground_truth_dir = config.DEFAULT_GROUND_TRUTH
    else:
        input_file = config.EVAL_VIDEO
        ground_truth_dir = config.GROUND_TRUTH
    scores = model_engine.evaluate(input_file, ground_truth_dir)
    logging.info(f"Scores: {scores}")
    return scores


def serve(json_data: Optional[Dict] = None, save_result: bool = False):
    """
    Wrapper function for serving of the Summarizer model engine

    Parameters:
        json_data: Input JSON for serving
        save_result: whether to save result locally or not (True for functional testing)

    Returns:
        API response for the input JSON

    """
    model_engine = network(json_data=json_data)
    response = model_engine.serve(save_result)
    return response


def package_test(test_mode: str):
    """
    Package test for functional testing of the Framify package

    Parameters:
        test_mode: mode of the functional testing(all, evaluate or serve)

    """
    if test_mode == "all":
        logging.debug("Testing all the components of the package..")

        logging.debug("Testing Evaluation Component..")
        evaluate()
        logging.debug("Evaluation Component test passed..")

        logging.debug("Testing Serving Component")
        serve(save_result=True)
        logging.debug("Serving component test passed..")

    elif test_mode == "serve":
        logging.debug("Testing Serving component")
        serve(save_result=True)
        logging.debug("Serving component test passed..")

    elif test_mode == "eval" or test_mode == "evaluate":
        logging.debug("Testing Evaluation component")
        evaluate()
        logging.debug("Evaluating Component passed..")
    else:
        logging.exception(
            "Invalid argument for func_test. Argument should be from [eval, serve]"
        )


def main():
    """
    Main function for the Framify(Keyframe extractor) package

    """
    cmd_args = arg_parser()
    if cmd_args.func_test:
        # Running the functional test
        logging.debug("Initializing Framify package's functional testing..")
        package_test(cmd_args.func_test)

    else:
        if cmd_args.mode == "eval" or cmd_args.mode == "evaluate":
            # Evaluating the model
            logging.debug("Initializing Framify model for Evaluation..")
            model_engine = network()
            model_engine.evaluate()

        elif cmd_args.mode == "serve":
            # Serving process of the model
            logging.debug("Initializing Framify model for Serving..")
            if cmd_args.file_path:
                json_data = utils_tools.load_json(cmd_args.file_path)
            else:
                json_data = None
            model_engine = network(json_data)
            response = model_engine.serve()
            logging.debug(f"The response from the Model : {response}")

        elif cmd_args.mode == "package_test":

            if not cmd_args.func_test:
                logging.exception(
                    " Pass an argument for func_test to run the package for testing.."
                )
            else:
                logging.debug(
                    "Instantiating Framify Engine package's functional testing.."
                )
                package_test(cmd_args.func_test)

        else:
            logging.exception(
                "Invalid Argument mode argument passed. Argument should be from [eval, serve]",
                exc_info=True,
            )


if __name__ == "__main__":
    main()
