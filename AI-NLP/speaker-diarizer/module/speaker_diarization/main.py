"""
Highest level python script for the Pratilipi's Speaker Diarizer. This script can be used to run the package in the
standalone mode which enables the Sentiment classifier to be trained, evaluated, served and tested agnostic of the
application interface code.

"""
import argparse
import logging
import os
# Appending the Package path to the system PATH variable
import sys

sys.path.append("..")

from speaker_diarization import config
from speaker_diarization.ml_engine import engine
from speaker_diarization.utils import csv_validator

# Adding the StreamHandler to print the logfile output to the stderr stream.
logging.getLogger().addHandler(logging.StreamHandler())
logging.basicConfig(
    filename=config.LOG_RESULTS_DIR + config.LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
)

# Flush the file, in case multiple runs happen in the same minute.
with open(os.path.join(config.LOG_RESULTS_DIR + config.LOG_FILE), "w+"):
    pass


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="train, eval, serve and package_test mode to run the sentiment engine",
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
    args = parser.parse_args()

    return args


def test_train():
    # TBA when the MLOPs integration is established.
    pass


def test_eval():
    # TBA when the MLOPs integration is established.
    pass


def test_serve():
    # In test serve mode, the system logs the output to the log file.
    model = engine.DiarizeEngine()
    response = model.serve(config.TEST_FILE, mode="save")
    logging.debug(f"The response from the model stored at : {response}")
    return response


def package_test(test):
    """

    :param test: The mode in which we want to test the package, string input.
    """

    if test == "all":
        logging.debug("Testing all the components of the package..")
        logging.debug("Testing Training Component..")
        test_train()
        logging.debug("Training Component not supported currently..")

        logging.debug("Testing Serving Component..")
        resp_csv = test_serve()

        if csv_validator.validate(resp_csv):
            logging.debug("Serving Component Test Passed..")
        else:
            logging.debug("Serving Component Test Failed..")

        logging.debug("Testing Evaluation Component..")

        test_eval()
        logging.debug("Evaluation Component not supported currently...")

    elif test == "train":
        logging.debug("Training Component - NOT SUPPORTED AT THE PRESENT")
        test_train()

    elif test == "serve":
        logging.debug("Testing Serving Component - standalone mode..")
        resp_csv = test_serve()
        if csv_validator.validate(resp_csv):
            logging.debug("Serving Component Test Passed..")
        else:
            logging.debug(
                "Serving Component Test Failed, Generated CSV doesnot match the GT CSV.."
            )

    elif test == "evaluate":
        logging.debug("Testing Evaluation Component - NOT SUPPORTED AT THE PRESENT")
        test_eval()
    else:
        logging.exception("Invalid functional test mode", exc_info=True)


def main():
    cmd_args = arg_parser()

    if cmd_args.mode == "train":
        logging.debug("Instantiating model Sentiment Engine for training..")
        model = engine.DiarizeEngine()
        model.train_uisrnn()
        logging.debug("Training component NOT Supported at the Present..")

    elif cmd_args.mode == "eval":
        logging.debug("Instantiating model Sentiment Engine for Evaluation..")
        logging.debug("Evaluation component NOT Supported at the present..")

    elif cmd_args.mode == "serve":

        logging.debug("Instantiating model Sentiment Engine for Serving..")
        model = engine.DiarizeEngine()
        response = model.serve(config.TEST_FILE, mode="save")
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
