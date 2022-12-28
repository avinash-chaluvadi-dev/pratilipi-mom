"""
Highest level python script for the Pratilipi's Sentiment Classifier. This script can be used to run the package in the
standalone mode which enables the Sentiment classifier to be trained, evaluated, served and tested agnostic of the
application interface code.

"""
import argparse
import logging
import os

from torch.utils.data import DataLoader

from . import config
from .ml_engine.data_loader import ClassifierDataset
from .ml_engine.engine import SentimentEngine
from .ml_engine.model import SentimentBackbone

# Adding the StreamHandler to print the logfile output to the stderr stream.
if not config.USE_EFS:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.basicConfig(
        filename=os.path.join(config.OUTPUT_LOG, config.LOG_FILE),
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )
    # Flush the file, in case multiple runs happen in the same minute.
    with open(os.path.join(config.OUTPUT_LOG, config.LOG_FILE), "w+"):
        pass

else:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )


def arg_parser():
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
        "--training",
        help="flag to be used to determine, whether you want to train from the scratch or"
        "resume",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    return args


def network(args):
    """

    :param mode: Instantiate the Sentiment engine using a particular mode, train, eval or serve.
    :return: return the Sentiment engine object
    """

    # Instantiate Model Creator Class, load the model architecture and the weights, if exists.
    # For scratch training set SentimentBackbone(config.TRANSFORMER_MODEL)
    # For resuming trainnig from a checkpoint SentimentBackbone(config.BASE_FINETUNED_CLASSIFIER)
    dataset = None
    logging.debug("Invoking Backbone Network Instantiator function.. ")
    if args.training == "resume":
        backbone_model = SentimentBackbone(config.BASE_FINETUNED_CLASSIFIER)
    else:
        backbone_model = SentimentBackbone(config.BASE_FINETUNED_CLASSIFIER)

    if args.mode == "train":
        # Instantiate the dataset object, parsing through the dataset folder.
        # This read the dataset from the config.py -> "TRAIN_JSON".
        dataset = ClassifierDataset(backbone_model, load_mode=args.mode)

    elif args.mode == "serve":
        # Instantiate the dataset object, parsing through the dataset folder.
        # This mode takes input a JSON and returns the response
        test_input = config.SERVE_INPUT
        dataset = ClassifierDataset(
            backbone_model, load_mode=args.mode, json_data=test_input
        )

    elif args.mode == "eval":
        # Instantiate the dataset object, parsing through the dataset folder.
        # This read the dataset from the config.py -> "EVAL_JSON".
        dataset = ClassifierDataset(backbone_model, load_mode=args.mode)

    else:
        logging.exception("Invalid mode for network creation..", exc_info=True)

    # Instantiate the Data-loader.
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Instantiate the Sentiment Engine.
    model_engine = SentimentEngine(model=backbone_model, data_loader=data_loader)

    return model_engine


def test_train(backbone_model):
    """
    Method to test the training functionality of the Sentiment Engine. Using the function_test.json file.
    """

    # Test the training component
    # This read the dataset from the config.py -> "function_test.json"
    dataset = ClassifierDataset(backbone_model, load_mode="train")

    # Instantiate the Data-loader.
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Instantiate the Sentiment Engine.
    model_engine = SentimentEngine(model=backbone_model, data_loader=data_loader)

    model_engine.train()


def test_serve(backbone_model):
    """
    Method to test the serving functionality of the Sentiment Engine. Using the function_test.json file.
    """

    # Test the training component
    # This read the dataset from the config.py -> "function_test.json"
    dataset = ClassifierDataset(backbone_model, load_mode="serve")

    # Instantiate the Data-loader.
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Instantiate the Sentiment Engine.
    model_engine = SentimentEngine(model=backbone_model, data_loader=data_loader)

    _ = model_engine.serve(save_result=True)


def test_evaluate():
    pass


def package_test(test):
    """

    :param test: The mode in which we want to test the package, string input.
    """

    if test == "all":
        # Instantiate Model Creator Class, load the model architecture and the weights.
        backbone_model = SentimentBackbone()
        logging.debug("Testing all the components of the package..")
        logging.debug("Testing Training Component..")
        test_train(backbone_model)
        logging.debug("Training Component Test Passed..")

        logging.debug("Testing Serving Component..")
        test_serve(backbone_model)
        logging.debug("Serving Component Test Passed..")

        # Evaluation component needs to written.
        # logging.debug("Evaluate training component..")
        # test_evaluate()
        # logging.debug("Evaluate Component Test Passed..")

    elif test == "train":
        backbone_model = SentimentBackbone()
        logging.debug("Testing Training Component - standalone mode..")
        test_train(backbone_model)
        logging.debug("Training Component Test Passed..")

    elif test == "serve":
        backbone_model = SentimentBackbone()
        logging.debug("Testing Serving Component - standalone mode..")
        test_serve(backbone_model)
        logging.debug("Serving Component Test Passed..")

    elif test == "evaluate":
        logging.debug("Testing Evaluation Component - NOT SUPPORTED AT THE PRESENT")
        test_evaluate()
    else:
        logging.exception("Invalid functional test mode", exc_info=True)


def main():
    cmd_args = arg_parser()

    if cmd_args.func_test:
        # Check the functional test
        logging.debug("Instantiating Sentiment Engine package's functional testing..")
        package_test(cmd_args.func_test)

    else:

        if cmd_args.mode == "train":
            logging.debug("Instantiating model Sentiment Engine for training..")
            model = network(cmd_args)
            model.train(save_result=True)

        elif cmd_args.mode == "eval":
            logging.debug("Instantiating model Sentiment Engine for Evaluation..")
            logging.debug("Evaluation component NOT Supported at the present..")
            # model = network(cmd_args.mode)
            # model.evaluate()

        elif cmd_args.mode == "serve":

            logging.debug("Instantiating model Sentiment Engine for Serving..")
            model = network(cmd_args)
            response = model.serve()
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
