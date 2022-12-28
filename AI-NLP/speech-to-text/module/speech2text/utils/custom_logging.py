import os
import logging
from datetime import datetime
from module.speech2text import config


def path_exists(path):
    return os.path.exists(path)


def get_logger():
    if not path_exists(config.OUTPUT_LOG):
        os.makedirs(config.OUTPUT_LOG)
    log_file = datetime.now().strftime("execution_%H-%M-%d-%m-%Y.log")
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s"
        )
        file_handler = logging.FileHandler(os.path.join(config.OUTPUT_LOG, log_file))
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


if __name__ == "__main__":
    log = get_logger()
    log.warning("This is warning")
