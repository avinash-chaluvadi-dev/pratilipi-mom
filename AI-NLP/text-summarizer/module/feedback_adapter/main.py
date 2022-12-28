import logging

from . import config
from .modules.annotator import TransAnnotator, annot_validator
from .modules.db_connector import DBConnector
from .modules.rectifier import TransRectifier, rectifier_validator

# Adding the StreamHandler to logging.debug the logfile output to the stderr stream.
logging.getLogger().addHandler(logging.StreamHandler())
if not config.USE_EFS:
    logging.basicConfig(
        filename=os.path.join(config.OUTPUT_LOG, config.LOG_FILE),
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )

if __name__ == "__main__":

    # Database Connection
    db_obj = DBConnector(config.TEST_DICT, config.RULES_DICT, config.LABEL_DICT)

    # Load the Corresponding artifacts.
    logging.debug("Loading Transcripts ...")
    transcripts = db_obj.load_transcripts()
    logging.debug("Loading Rule Dictionary ...")
    rule_dict = db_obj.load_rule_dict()
    logging.debug("Loading Label Dictionary ...")
    label_dict = db_obj.load_label_dict()

    # Instantiate the rectifier perform the cleanup.
    logging.debug("Rectifying Transcripts ...")
    trans_rect = TransRectifier(transcripts["Audited Transcripts"], rule_dict)
    generated_rect = trans_rect.rectify()
    logging.debug("Transcription Rectification Completed ...")

    # Rectification Validation
    logging.debug("Testing the System Rectified Transcripts ...")
    rectifier_validator(
        transcripts["Cleaned Transcripts"], generated_rect["Cleaned Transcripts"]
    )
    logging.debug("Rectification Component Testing Completed ...")

    # Instantiate the annotator and perform the tagging.
    logging.debug("Annotating Transcripts ...")
    trans_annot = TransAnnotator(transcripts["Audited Transcripts"])
    generated_label = trans_annot.tagger()

    # Annotation Validation
    logging.debug("Testing the System Annotated Transcripts ...")
    annot_validator(
        transcripts["Annotation Labels"], generated_label["Annotation Labels"]
    )
    logging.debug("Annotation Component Testing Completed ...")
    logging.debug("Transcription Annotations Parsing Completed ...")

    # Merge the two dicts together
    logging.debug("Merge Rectification and Annotation results ...")
    logging.debug("Adaptor Output :")
    generated_rect.update(generated_label)

    # Push the cleaned up transcripts to the DB.
    logging.debug("Push the Annotation Adapter Output to the Storage ...")
    db_obj.push_to_db(config.RESULT_STORE, generated_rect)
    logging.debug("Store Operation Complete ...")

    # Currently local storage to store the output.
    # DB operation needs to be added.
    db_obj.push_to_db(config.RESULT_STORE, generated_rect)
