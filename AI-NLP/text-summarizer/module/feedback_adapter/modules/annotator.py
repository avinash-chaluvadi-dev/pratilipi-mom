import ast
import logging

from feedback_adapter import config

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


class TransAnnotator:
    """
    This class should be able to fetch the labels dictionary from the audited transcripts and put that in the cleaned-up
    JSON to be used to populate the detailed view screen and the MoM Screen.

    Reference Data Structure of the Tags added Transcripts ->
    [SOT]<>Piece [#] Please [/#] </> update your <> jeera [#] Jira [/#]</> story with comments so that [] a [/] <> meet [#] Amit [/#] </>
    could have a <> hook [#] look [/#]</>.[EOT]

    Reference Data structure to parse labels:
            {
                “Sentiment” : “NEU”,
                “Entity” : { "Words" : [], "Type": [] },
                “Deadline” : “ ”,
                “Escalation”: “ ”,
                “Help”: “ ”,
                “Question”: “ ”,
            }

        # Data structure of the label_dictionary.
        # Generic format -> {"Model_name" : [name of the labels]}

        dict_labels =
        {
            "Sentiment" : ["Neu", "Pos", "Neg"],
            "Entity": ["Name", "Tool", "Anthem Tool", "Technology"],
            "Deadline": ["Yes", "No"],
            "Escalation": ["Yes", "No"],
            "Help": ["Yes", "No"],
            "Question": ["Yes", "No"]
        }

    """

    def __init__(self, transcripts=None):
        self.transcripts = transcripts

    def tagger(self):
        """
        Highest level label tagging method, to process the batch of transcripts with tags added.

        """
        labels_list = []

        for transcript in self.transcripts:
            # Get the Labels dictionary out from the transcripts.
            start = transcript.find("{")
            end = transcript.rfind("}") + 1
            labels = transcript[start:end]
            labels = ast.literal_eval(labels)
            labels_list.append(labels)
        return {"Annotation Labels": labels_list}


def annot_validator(gt_labels, generated_labels):

    if gt_labels == generated_labels:
        logging.debug(
            "The Labels are correctly generated : Annotation Validation Test Passed ..."
        )
    else:
        logging.debug(" Annotation Validation Test Failed ...")
