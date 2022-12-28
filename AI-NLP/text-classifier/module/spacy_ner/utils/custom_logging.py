import logging
import os
from datetime import datetime

from .. import config
from . import utils_tools


def get_logger():
    """
    Custom Logger to debug, log the information, warnings and exceptions
    """
    if not config.USE_EFS and not utils_tools.path_exists(config.OUTPUT_LOG):
        os.makedirs(config.OUTPUT_LOG)
        log_file = datetime.now().strftime("execution_%H-%M-%d-%m-%Y.log")
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s"
        )
        if not config.USE_EFS:
            file_handler = logging.FileHandler(
                os.path.join(config.OUTPUT_LOG, log_file)
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
    return logger
