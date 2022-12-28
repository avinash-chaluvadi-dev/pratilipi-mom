import os
import sys
import runpy
import tensorflow as tf
from os.path import dirname as up

sys.path.insert(1, up(up(up(__file__))))

from module.speech2text.ml_engine import engine
from module.speech2text.utils import custom_logging,utils_tools



def model_serve(test_input):
    logger = custom_logging.get_logger()
    config_module = runpy.run_path("module/speech2text/config.py", init_globals={"tf": tf})
    base_config = config_module.get("base_params", None)

    logger.debug(base_config)
    if base_config is None:
        logger.exception(
            "base_config dictionary has to be " "defined in the config file"
        )
        raise ValueError(
            "base_config dictionary has to be " "defined in the config file"
        )
    base_model = config_module.get("base_model", None)
    if base_model is None:
        logger.exception("base_config class has to be defined in the config file")
        raise ValueError("base_config class has to be defined in the config file")
    hvd = utils_tools.get_hvd(base_config)
    load_model = base_config.get("load_model", None)
    restore_best_checkpoint = base_config.get("restore_best_checkpoint", False)
    base_ckpt_dir = utils_tools.check_base_model_logdir(
        base_logdir=load_model, restore_best_checkpoint=restore_best_checkpoint
    )
    base_config["load_model"] = base_ckpt_dir
    checkpoint = utils_tools.check_logdir(
        base_config=base_config, restore_best_checkpoint=restore_best_checkpoint
    )
    with tf.Graph().as_default():
        model = utils_tools.create_model(
            mode="infer",
            base_config=base_config,
            config_module=config_module,
            base_model=base_model,
            hvd=hvd,
            checkpoint=checkpoint,
            json_data=test_input,
        )
        logger.debug("Instantiating model Frame Classifier Engine for Serving..")
        logger.debug(model)
        model_engine = engine.Speech2TextEngine()
        if hvd is None or hvd.rank() == 0:
            logger.debug("Loading model from {}".format(checkpoint))
            output = model_engine.infer(model, checkpoint)
            output = utils_tools.create_output_json(json_data=test_input, response=output)
            return output
