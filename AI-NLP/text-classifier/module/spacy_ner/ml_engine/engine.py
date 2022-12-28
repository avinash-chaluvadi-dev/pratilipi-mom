import os
import random
from collections import defaultdict

import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from tqdm import tqdm

from .. import config
from ..utils import custom_logging, utils_tools


class NEREngine:
    """
    NER engine class: This class to encapsulate the train, serving and evaluating function of the
    NER Model
    """

    def __init__(
        self,
        model,
        save_model=False,
        train_data_loader=None,
        eval_data_loader=None,
        serve_data_loader=None,
    ):
        self.model = model.nlp
        self.ner = model.ner
        self.save_model = save_model
        self.logger = custom_logging.get_logger()
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.serve_data_loader = serve_data_loader

    def train(self, num_epochs=None):
        if num_epochs is None:
            epochs = config.N_ITER
        else:
            epochs = num_epochs
        try:
            traindata = self.train_data_loader
            for _, annotations in traindata:
                for ent in annotations.get(config.params["train"]["entities_key"]):
                    self.ner.add_label(ent[2])

            other_pipes = [pipe for pipe in self.model.pipe_names if pipe != "ner"]
            with self.model.disable_pipes(*other_pipes):  # only train NER
                optimizer = self.model.begin_training()
                self.logger.debug(
                    "Starting the Training Loop .."
                )  # Training loop start
                for epoch in range(epochs):
                    self.logger.debug(f"[INFO] Epoch {epoch + 1} Started..")
                    random.shuffle(traindata)
                    losses = {}
                    for text, annotations in tqdm(traindata):
                        self.model.update(
                            [text],
                            [annotations],
                            drop=config.DROPOUT,
                            sgd=optimizer,
                            losses=losses,
                        )
                    self.logger.debug(f"Epoch: {epoch+1}: {losses}")

            # save model
            if self.save_model:
                utils_tools.save_model(self.model)

        except (RuntimeError, MemoryError, ValueError, TypeError):
            self.logger.exception("Training Exception Occurred")

    def eval(self):
        try:
            eval_scores = {}
            self.logger.debug("Starting the Evaluation ..")  # Evaluation loop start

            ner_model = spacy.load(os.path.join(config.MODELS_DIR, config.BEST_MODEL))
            scorer = Scorer()
            for input_, annot in self.eval_data_loader:
                doc_gold_text = ner_model.make_doc(input_)
                gold = GoldParse(
                    doc_gold_text,
                    entities=annot[config.params["eval"]["entities_key"]],
                )
                pred_value = ner_model(input_)
                scorer.score(pred_value, gold)

            eval_scores = scorer.scores
            eval_st = f"Evaluation scores: {eval_scores}"
            self.logger.debug(f"{eval_st}")
            return eval_scores

        except (RuntimeError, MemoryError, ValueError, TypeError):
            self.logger.exception("Evaluation Exception Occurred")

    def serve(self):
        try:
            predictions = []
            for transcript in self.serve_data_loader:
                ent_with_scores = get_ner_scores(transcript, self.model)
                predictions.append(ent_with_scores)
            response = {
                "ner_result": {
                    "status": "Success",
                    "details": predictions,
                }
            }
            return response
        except RuntimeError:
            self.logger.exception(
                "Exception encountered while serving the NER Engine",
                exc_info=True,
            )
            response = {
                "ner_result": {
                    "status": "Error",
                    "details": f"{traceback.format_exc()}",
                }
            }
            return response


def get_ner_scores(transcript, ner_model):
    with ner_model.disable_pipes("ner"):
        doc = ner_model(transcript[config.params["serve"]["text_key"]])

    token_list = [token.idx for token in doc]
    # beam_width - Number of alternate analyses to consider. More is slower, and not necessarily better -- you need to experiment on your problem.
    # beam_density - This clips solutions at each step. We multiply the score of the top-ranked action by this value, and use the result as a threshold. This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.

    beams = (
        ner_model.entity.beam_parse([doc], beam_width=16, beam_density=0.0001)
        if len(token_list)
        else []
    )

    entity_scores = defaultdict(float)
    for beam in beams:
        for score, ents in ner_model.entity.moves.get_beam_parses(beam):
            for start, end, label in ents:
                entity_scores[
                    (
                        token_list[start],
                        token_list[start] + len(str(doc[start:end])),
                        label,
                        str(doc[start:end]),
                    )
                ] += score

    final_entity, confidence_scores = get_final_entities(entity_scores)

    transcript[config.params["serve"]["entities_key"]] = final_entity
    transcript[config.params["serve"]["confidence_score_key"]] = confidence_scores
    return transcript


def get_final_entities(entity_scores):
    final_entity_scores = {}
    for key in entity_scores:
        if entity_scores[key] < config.CONFIDENCE_THRESHOLD:
            continue
        if (key[0], key[1]) not in final_entity_scores.keys():
            final_entity_scores[(key[0], key[1])] = (
                key[2],
                key[3],
                entity_scores[key],
            )
        else:
            if entity_scores[key] > final_entity_scores[(key[0], key[1])][2]:
                final_entity_scores[(key[0], key[1])] = (
                    key[2],
                    key[3],
                    entity_scores[key],
                )
    confidence_scores = []
    words = []
    types = []
    for key in final_entity_scores:
        words.append(final_entity_scores[key][1])
        types.append(final_entity_scores[key][0])
        confidence_scores.append(final_entity_scores[key][2])
    final_entity = {"words": words, "type": types}
    return final_entity, confidence_scores
