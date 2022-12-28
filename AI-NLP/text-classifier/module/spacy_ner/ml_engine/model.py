import spacy

from .. import config
from ..utils import custom_logging


class SpacyNER:
    logger = custom_logging.get_logger()

    def __init__(self, nlp, ner, framework=None):
        self.framework = framework
        self.nlp = nlp
        self.ner = ner

    @classmethod
    def from_spacy_model(cls, model=None):
        if model is not None:
            nlp = config.model

        else:
            nlp = spacy.blank("en")
            cls.logger.info("Created blank 'en' model")

        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner, last=True)
        else:
            ner = config.ner

        return cls(nlp=nlp, ner=ner)
