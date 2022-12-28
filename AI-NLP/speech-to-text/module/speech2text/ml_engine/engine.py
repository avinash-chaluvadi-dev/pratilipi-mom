from __future__ import unicode_literals
from __future__ import absolute_import, division, print_function
import time
import traceback

import numpy as np
import tensorflow as tf
from six.moves import range
from module.speech2text.utils import utils_tools
from module.speech2text.utils import custom_logging
from tensorflow.python import debug as tf_debug
from module.speech2text.utils import hooks as tf_hooks
from module.speech2text.seq2seq.data import WKTDataLayer


class Speech2TextEngine:
    def __init__(self, debug_port=None, custom_hooks=None):
        self.debug_port = debug_port
        self.custom_hooks = custom_hooks
        self.logger = custom_logging.get_logger()

    def train(self, train_model, eval_model=None):
        if eval_model is not None and "eval_steps" not in eval_model.params:
            self.logger.exception(
                "eval_steps parameter has to be specified " "if eval_model is provided"
            )
        hvd = train_model.hvd
        if hvd:
            master_worker = hvd.rank() == 0
        else:
            master_worker = True

        # Initializing session parameters
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        # pylint: disable=no-member
        sess_config.gpu_options.allow_growth = True
        if hvd is not None:
            # pylint: disable=no-member
            sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

        if train_model.params.get("use_xla_jit", False):
            sess_config.graph_options.optimizer_options.global_jit_level = (
                tf.OptimizerOptions.ON_1
            )

        # defining necessary hooks
        hooks = [tf.train.StopAtStepHook(last_step=train_model.last_step)]
        if self.custom_hooks:
            for custom_hook in self.custom_hooks:
                hooks.append(
                    custom_hook(train_model=train_model, eval_model=eval_model)
                )

        if hvd is not None:
            hooks.append(tf_hooks.BroadcastGlobalVariablesHook(0))

        if master_worker:
            checkpoint_dir = train_model.params["logdir"]
            load_model_dir = train_model.params["load_model"]
        else:
            checkpoint_dir = None
            load_model_dir = None

        if eval_model is not None:
            # noinspection PyTypeChecker
            hooks.append(
                hooks.RunEvaluationHook(
                    every_steps=eval_model.params["eval_steps"],
                    model=eval_model,
                    last_step=train_model.last_step,
                    print_ppl=isinstance(eval_model.get_data_layer(), WKTDataLayer),
                ),
            )

        if master_worker:
            if train_model.params["save_checkpoint_steps"] is not None:
                # noinspection PyTypeChecker
                saver = tf.train.Saver(
                    save_relative_paths=True,
                    max_to_keep=train_model.params["num_checkpoints"],
                )
                hooks.append(
                    tf.train.CheckpointSaverHook(
                        checkpoint_dir,
                        saver=saver,
                        save_steps=train_model.params["save_checkpoint_steps"],
                    )
                )
            if train_model.params["print_loss_steps"] is not None:
                # noinspection PyTypeChecker
                hooks.append(
                    tf_hooks.PrintLossAndTimeHook(
                        every_steps=train_model.params["print_loss_steps"],
                        model=train_model,
                        print_ppl=isinstance(
                            train_model.get_data_layer(), WKTDataLayer
                        ),
                    )
                )
            if train_model.params["print_samples_steps"] is not None:
                # noinspection PyTypeChecker
                hooks.append(
                    tf_hooks.PrintSamplesHook(
                        every_steps=train_model.params["print_samples_steps"],
                        model=train_model,
                    )
                )

        total_time = 0.0
        bench_start = train_model.params.get("bench_start", 10)

        if self.debug_port:
            hooks.append(
                tf_debug.TensorBoardDebugHook("localhost:{}".format(self.debug_port))
            )

        if train_model.on_horovod:
            init_data_layer = train_model.get_data_layer().iterator.initializer
        else:
            init_data_layer = tf.group(
                [
                    train_model.get_data_layer(i).iterator.initializer
                    for i in range(train_model.num_gpus)
                ]
            )

        # We restore only if the user provides load_model_dir. load_model_dir is the
        # directory containing the checkpoint we want to load partial or all weights
        # from.. Useful for transer learning or if we do not want to overwrite our
        # checkpoint.
        restoring = load_model_dir and not tf.train.latest_checkpoint(checkpoint_dir)
        if restoring:
            vars_in_checkpoint = {}
            for var_name, var_shape in tf.train.list_variables(load_model_dir):
                vars_in_checkpoint[var_name] = var_shape

            print("VARS_IN_CHECKPOINT:")
            print(vars_in_checkpoint)

            vars_to_load = []
            for var in tf.global_variables():
                var_name = var.name.split(":")[0]
                if var_name in vars_in_checkpoint:
                    if (
                        var.shape == vars_in_checkpoint[var_name]
                        and "global_step" not in var_name
                    ):
                        vars_to_load.append(var)

            print("VARS_TO_LOAD:")
            for var in vars_to_load:
                print(var)

            load_model_fn = tf.contrib.framework.assign_from_checkpoint_fn(
                tf.train.latest_checkpoint(load_model_dir), vars_to_load
            )
            scaffold = tf.train.Scaffold(
                local_init_op=tf.group(
                    tf.local_variables_initializer(), init_data_layer
                ),
                init_fn=lambda scaffold_self, sess: load_model_fn(sess),
            )

        else:
            scaffold = tf.train.Scaffold(
                local_init_op=tf.group(
                    tf.local_variables_initializer(), init_data_layer
                )
            )
        fetches = [train_model.train_op]
        try:
            total_objects = 0.0
            # on horovod num_gpus is 1
            for worker_id in range(train_model.num_gpus):
                fetches.append(train_model.get_num_objects_per_step(worker_id))
        except NotImplementedError:
            self.logger.exception(
                "Can't compute number of objects per step, since "
                "train model does not define get_num_objects_per_step method."
            )
            self.logger.exception(
                "WARNING: Can't compute number of objects per step, since "
                "train model does not define get_num_objects_per_step method."
            )

        # starting training
        sess = tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            checkpoint_dir=checkpoint_dir,
            save_summaries_steps=train_model.params["save_summaries_steps"],
            config=sess_config,
            save_checkpoint_secs=None,
            log_step_count_steps=train_model.params["save_summaries_steps"],
            stop_grace_period_secs=300,
            hooks=hooks,
        )
        step = 0
        num_bench_updates = 0
        while True:
            if sess.should_stop():
                break
            tm = time.time()
            try:
                feed_dict = {}
                iter_size = train_model.params.get("iter_size", 1)
                if iter_size > 1:
                    feed_dict[train_model.skip_update_ph] = step % iter_size != 0
                if step % iter_size == 0:
                    if step >= bench_start:
                        num_bench_updates += 1
                    fetches_vals = sess.run(fetches, feed_dict)
                else:
                    # necessary to skip "no-update" steps when iter_size > 1
                    def run_with_no_hooks(step_context):
                        return step_context.session.run(fetches, feed_dict)

                    fetches_vals = sess.run_step_fn(run_with_no_hooks)
            except tf.errors.OutOfRangeError:
                break
            if step >= bench_start:
                total_time += time.time() - tm
                if len(fetches) > 1:
                    for i in range(train_model.num_gpus):
                        total_objects += np.sum(fetches_vals[i + 1])
                    if train_model.params["print_bench_info_steps"] is not None:
                        if step % train_model.params["print_bench_info_steps"] == 0:
                            total_objects_cur = utils_tools.collect_if_horovod(
                                total_objects, hvd, mode="sum"
                            )
                            if master_worker:
                                avg_objects = 1.0 * total_objects_cur / total_time
                                self.logger.debug(
                                    "Avg objects per second: {:.3f}".format(avg_objects)
                                )
            step += 1
        sess.close()

        if len(fetches) > 1:
            total_objects = utils_tools.collect_if_horovod(
                total_objects, hvd, mode="sum"
            )

        if master_worker:
            self.logger.info("Finished training")
            if step > bench_start:
                avg_time = 1.0 * total_time / num_bench_updates
                self.logger.debug("Avg time per step: {:.3f}s".format(avg_time))
                if len(fetches) > 1:
                    avg_objects = 1.0 * total_objects / total_time
                    self.logger.exception(
                        "Avg objects per second: {:.3f}".format(avg_objects)
                    )
            else:
                self.logger.debug("Not enough steps for benchmarking")

    def eval(self, model, checkpoint):
        results_per_batch = utils_tools.restore_and_get_results(
            model, checkpoint, mode="eval"
        )
        if not model.on_horovod or model.hvd.rank() == 0:
            eval_dict = model.finalize_evaluation(results_per_batch)
            self.logger.info("Finished evaluation")
            return eval_dict
        return None

    def infer(self, model, checkpoint, output_file=None):
        try:
            results_per_batch = utils_tools.restore_and_get_results(
                model, checkpoint, mode="infer"
            )
            if not model.on_horovod or model.hvd.rank() == 0:
                output = model.finalize_inference(results_per_batch, output_file)
                # print(output)
                self.logger.info("Finished Inference")
                return output
        except RuntimeError as e:
            return {"speech_to_text_output": {"status": "fail", "details": f"{traceback.format(e)}"}}
