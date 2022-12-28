from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
import tensorflow as tf
from module.speech2text import config
from module.speech2text.ml_engine import engine
from module.speech2text.utils import utils_tools, custom_logging

if hasattr(tf.compat, "v1"):
    tf.compat.v1.disable_eager_execution()


def run(**kwargs):
    hvd = kwargs.get("hvd")
    args = kwargs.get("args")
    mode = kwargs.get("mode")
    model = kwargs.get("model")
    checkpoint = kwargs.get("checkpoint")
    base_ckpt_dir = kwargs.get("base_ckpt_dir")
    logger = custom_logging.get_logger()
    if mode == "train":
        if hvd is None or hvd.rank() == 0:
            if checkpoint is None or args.benchmark:
                if base_ckpt_dir:
                    logger.debug("Starting training from the base model")
                else:
                    logger.debug("Starting training from scratch")
            else:
                logger.debug(
                    "Restored checkpoint from {}. Resuming training".format(checkpoint),
                )
        model_engine = engine.Speech2TextEngine()
        model_engine.train(train_model=model, eval_model=None)
    if mode == "eval":
        if hvd is None or hvd.rank() == 0:
            utils_tools.deco_print("Loading model from {}".format(checkpoint))
            model_engine = engine.Speech2TextEngine()
            eval_metrics_dict = model_engine.eval(model, checkpoint)
            return eval_metrics_dict

    if mode == "infer":
        if hvd is None or hvd.rank() == 0:
            utils_tools.deco_print("Loading model from {}".format(checkpoint))
        model_engine = engine.Speech2TextEngine()
        if hvd is None or hvd.rank() == 0:
            logger.debug("Loading model from {}".format(checkpoint))
            output = model_engine.infer(model, checkpoint)
            return output


class PackageTest:
    def __init__(self, **kwargs):
        self.hvd = kwargs.get("hvd")
        self.args = kwargs.get("args")
        self.mode = kwargs.get("mode")
        self.logger = custom_logging.get_logger()
        self.checkpoint = kwargs.get("checkpoint")
        self.base_model = kwargs.get("base_model")
        self.base_config = kwargs.get("base_config")
        self.base_ckpt_dir = kwargs.get("base_ckpt_dir")
        self.config_module = kwargs.get("config_module")

    def test_train_component(self):
        model = utils_tools.create_model(
            mode="eval",
            args=self.args,
            base_config=self.base_config,
            config_module=self.config_module,
            base_model=self.base_model,
            hvd=self.hvd,
        )
        if self.hvd is None or self.hvd.rank() == 0:
            if self.checkpoint is None or self.args.benchmark:
                if self.base_ckpt_dir:
                    self.logger.debug("Starting training from the base model")
                else:
                    self.logger.debug("Starting training from scratch")
            else:
                self.logger.debug(
                    "Restored checkpoint from {}. Resuming training".format(self.checkpoint),
                )
        model_engine = engine.Speech2TextEngine()
        model_engine.train(train_model=model, eval_model=None)

    def test_eval_component(self):
        with tf.Graph().as_default():
            model = utils_tools.create_model(
                mode="eval",
                args=self.args,
                base_config=self.base_config,
                config_module=self.config_module,
                base_model=self.base_model,
                hvd=self.hvd,
                checkpoint=self.checkpoint,
            )
            if self.hvd is None or self.hvd.rank() == 0:
                utils_tools.deco_print("Loading model from {}".format(self.checkpoint))
            model_engine = engine.Speech2TextEngine()
            if self.hvd is None or self.hvd.rank() == 0:
                utils_tools.deco_print("Loading model from {}".format(self.checkpoint))
                eval_metrics_dict = model_engine.eval(model, self.checkpoint)
                return eval_metrics_dict

    def test_infer_component(self):
        with tf.Graph().as_default():
            model = utils_tools.create_model(
                mode="infer",
                args=self.args,
                base_config=self.base_config,
                config_module=self.config_module,
                base_model=self.base_model,
                hvd=self.hvd,
                checkpoint=self.checkpoint,
            )
            self.logger.debug(self.args.mode)
            self.args.mode = "infer"
            self.logger.debug(self.args.mode)
            if self.hvd is None or self.hvd.rank() == 0:
                utils_tools.deco_print("Loading model from {}".format(self.checkpoint))
            model_engine = engine.Speech2TextEngine()
            if self.hvd is None or self.hvd.rank() == 0:
                self.logger.debug("Loading model from {}".format(self.checkpoint))
                output = model_engine.infer(model, self.checkpoint)
                utils_tools.save_result(json_data=output, output_run=config.OUTPUT_RUN, out_json=config.OUT_JSON)
                return output


def main():
    logger = custom_logging.get_logger()
    args, base_config, base_model, config_module, hvd = utils_tools.get_base_config(
        sys.argv[1:]
    )
    logger.debug(args)
    load_model = base_config.get("load_model", None)
    restore_best_checkpoint = base_config.get("restore_best_checkpoint", False)
    base_ckpt_dir = utils_tools.check_base_model_logdir(
        base_logdir=load_model,
        args=args,
        restore_best_checkpoint=restore_best_checkpoint,
    )
    base_config["load_model"] = base_ckpt_dir
    checkpoint = utils_tools.check_logdir(
        args=args,
        base_config=base_config,
        restore_best_checkpoint=restore_best_checkpoint,
    )

    if args.enable_logs:
        if hvd is None or hvd.rank() == 0:
            old_stdout, old_stderr, stdout_log, stderr_log = utils_tools.create_logdir(
                args, base_config
            )
        base_config["logdir"] = os.path.join(base_config["logdir"], "logs")
    # with tf.Graph().as_default():
    #     # json_data = {
    #     #     "audio_chunks": [
    #     #         "dataset/hive_standup_20210804_audio_chunk_4828_19024_0.wav",
    #     #         "dataset/hive_standup_20210804_audio_chunk_21592_24736_1.wav",
    #     #         "dataset/hive_standup_20210804_audio_chunk_24736_44808_2.wav",
    #     #     ]
    #     # }
    #     model = utils_tools.create_model(
    #         mode=args.mode,
    #         args=args,
    #         base_config=base_config,
    #         config_module=config_module,
    #         base_model=base_model,
    #         hvd=hvd,
    #         checkpoint=checkpoint,
    #     )
    hooks = None
    if "train_params" in config_module and "hooks" in config_module["train_params"]:
        hooks = config_module["train_params"]["hooks"]
    if args.mode == "package_test":
        package_test_obj = PackageTest(
            hvd=hvd,
            args=args,
            mode=args.mode,
            base_model=base_model,
            checkpoint=checkpoint,
            base_config=base_config,
            config_module=config_module,
        )
        if args.func_test == "all":
            logger.debug("Testing all the components of the package..")
            logger.debug("Testing Training Component..")
            package_test_obj.test_train_component()
            logger.debug("Training Component Test Passed..")
            logger.debug("Testing Evaluation Component..")
            package_test_obj.test_eval_component()
            logger.debug("Evaluation Component Test Passed..")
            logger.debug("Testing Serving Component..")
            package_test_obj.test_infer_component()
            logger.debug("Serving Component Test Passed..")
            # package_test(test_mode=cmd_args.func_test)
        elif args.func_test == "train":
            logger.debug("Testing Training Component..")
            package_test_obj.test_train_component()
            logger.debug("Training Component Test Passed..")
        elif args.func_test == "eval":
            logger.debug("Testing Evaluation Component..")
            eval_metrics_dict = package_test_obj.test_eval_component()
            logger.info(eval_metrics_dict)
            logger.debug("Evaluation Component Test Passed..")
        elif args.func_test == "infer":
            logger.debug("Testing Inference Component..")
            output = package_test_obj.test_infer_component()
            logger.info(output)
            logger.debug("Inference Component Test Passed..")
        else:
            logger.exception("Invalid functional test mode")
    else:
        with tf.Graph().as_default():
            # json_data = {
            #     "audio_chunks": [
            #         "dataset/hive_standup_20210804_audio_chunk_4828_19024_0.wav",
            #         "dataset/hive_standup_20210804_audio_chunk_21592_24736_1.wav",
            #         "dataset/hive_standup_20210804_audio_chunk_24736_44808_2.wav",
            #     ]
            # }
            model = utils_tools.create_model(
                mode=args.mode,
                args=args,
                base_config=base_config,
                config_module=config_module,
                base_model=base_model,
                hvd=hvd,
                checkpoint=checkpoint,
            )
            if (
                "train_params" in config_module
                and "hooks" in config_module["train_params"]
            ):
                hooks = config_module["train_params"]["hooks"]

            if args.mode == "train":
                logger.debug("Instantiating model Speech To Text Engine for training..")
                run(
                    hvd=hvd,
                    args=args,
                    mode=args.mode,
                    model=model,
                    checkpoint=None,
                )
                logger.info("Finished Training!!")

            elif args.mode == "eval":
                logger.debug(
                    "Instantiating model Speech To Text Engine for evaluation.."
                )
                eval_metrics_dict = run(
                    hvd=hvd,
                    args=args,
                    mode=args.mode,
                    model=model,
                    checkpoint=checkpoint,
                )
                logger.info(eval_metrics_dict)

            elif args.mode == "infer":
                logger.debug("Instantiating Speech To Text Engine for Serving..")
                output = run(
                    hvd=hvd,
                    args=args,
                    mode=args.mode,
                    model=model,
                    checkpoint=checkpoint,
                )
                logger.info(output)
            else:
                logger.exception("Invalid mode Argument..", exc_info=True)


if __name__ == "__main__":
    main()
