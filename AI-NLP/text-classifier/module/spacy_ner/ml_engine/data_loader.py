import ast
import json
import os

import pandas as pd

from .. import config
from ..utils import custom_logging, utils_tools


class DatasetCreator:
    def __init__(self, train_file):
        self.logger = custom_logging.get_logger()
        self.train_file = train_file

    def create_train_data(self):
        try:
            dataframe = pd.DataFrame(
                columns=[
                    config.params["train"]["text_key"],
                    config.params["train"]["entities_key"],
                ]
            )
            files = os.listdir(config.JSONL_DIR)
            for file in files:
                if file.endswith(".jsonl"):
                    with open(os.path.join(config.JSONL_DIR, file)) as f:
                        json_obj = [json.loads(line) for line in f]
                        for labelled_txt in json_obj:
                            dataframe = dataframe.append(
                                {
                                    config.params["train"]["text_key"]: labelled_txt[
                                        config.JSONL_COLUMNS[0]
                                    ],
                                    config.params["train"][
                                        "entities_key"
                                    ]: labelled_txt[config.JSONL_COLUMNS[1]],
                                },
                                ignore_index=True,
                            )

            if not utils_tools.path_exists(
                path=os.path.join(config.LOAD_DIR, self.train_file)
            ):
                dataframe.to_csv(
                    os.path.join(config.LOAD_DIR, self.train_file), index=False
                )
                self.logger.exception("Saved training file...")
            else:
                self.logger.exception("Reading already existing training file...")
                dataframe = pd.read_csv(os.path.join(config.LOAD_DIR, self.train_file))
            return dataframe
        except MemoryError:
            self.logger.exception("Memory error occurred, ran out of memory")
        except (RuntimeError, ValueError, TypeError):
            self.logger.exception("Run Time Exception Occurred")


class NERDataset:
    """
    NERDataset class - to load the dataset.
    The loader supports the JSON and the csv format for parsing the input to the network.
    Note: only CSV format is supported for the training, while CSV and JSON are supported for the evaluation
    and testing.
    """

    logger = custom_logging.get_logger()

    def __init__(self, data, framework=None, mode=None):
        """This function is used to initialize the state of object(NERDataset).

        ARGS:
            texts(numpy.ndarray, list): Array of texts.
            labels(numpy.ndarray, list): Array of Entities.

        """
        self.mode = mode
        self.data = data
        self.framework = framework

    @classmethod
    def from_dataframe(cls, dataframe, mode, params=config.params["serve"]):
        data = []
        for ind in range(len(dataframe)):
            try:
                data.append(
                    (
                        dataframe[params["text_key"]][ind],
                        {
                            params["entities_key"]: ast.literal_eval(
                                dataframe[params["entities_key"]][ind]
                            )
                        },
                    )
                )
            except ValueError:
                data.append(
                    (
                        dataframe[params["text_key"]][ind],
                        {
                            params["entities_key"]: dataframe[params["entities_key"]][
                                ind
                            ]
                        },
                    )
                )
        return cls(data=data, mode=mode)

    @classmethod
    def from_json(cls, json_data, mode, params=config.params["serve"]):
        data = []
        if mode == "serve":
            data = json_data[params["response_key"]]
        else:
            for transcripts in json_data[params["response_key"]]:
                text = transcripts[params["text_key"]]
                entity_data = []
                for ent in transcripts[params["entities_key"]]:
                    entity_data.append([ent[0], ent[1], ent[2]])
                data.append((text, {params["entities_key"]: entity_data}))

        return cls(data=data, mode=mode)


if __name__ == "__main__":
    dataset_creator = DatasetCreator()
    dataframe = dataset_creator.create_train_data()
