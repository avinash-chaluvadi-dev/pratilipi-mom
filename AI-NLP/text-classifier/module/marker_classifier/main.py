import argparse

from .ml_engine import network
from .utils import custom_logging, utils_tools


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
        "--continue_learning",
        help="whether to continue learning",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--checkpoint", help="path of checkpoint to load the model", default=None
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


def main():
    """

    Main function for the Marker Classifier package

    """
    cmd_args = arg_parser()
    logger = custom_logging.get_logger()
    checkpoint, tokenizer = utils_tools.get_model_and_tokenizer(
        checkpoint=cmd_args.checkpoint, logger=logger
    )
    package_test_obj = network.PackageTest(
        mode=cmd_args.mode,
        func_test=cmd_args.func_test,
        save_model=cmd_args.save_model,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
    )

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
                "Instantiating Marker Classifier Engine for training and evaluation.."
            )
            network.run(
                mode=cmd_args.mode,
                save_model=cmd_args.save_model,
                continue_learning=cmd_args.continue_learning,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
            )

        elif cmd_args.mode == "train":
            logger.debug("Instantiating Marker Classifier Engine for training..")
            network.run(
                mode=cmd_args.mode,
                save_model=cmd_args.save_model,
                continue_learning=cmd_args.continue_learning,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
            )

        elif cmd_args.mode == "eval":
            logger.debug("Instantiating Marker Classifier Engine for evaluation..")
            network.run(
                mode=cmd_args.mode,
                save_model=cmd_args.save_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
            )

        elif cmd_args.mode == "serve":
            logger.debug("Instantiating Marker Classifier Engine for serving..")
            network.run(
                mode=cmd_args.mode,
                save_model=cmd_args.save_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
            )

        else:
            logger.exception("Invalid mode Argument..", exc_info=True)


if __name__ == "__main__":
    main()
