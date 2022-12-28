# Setup a local DB and do the operation for the batch of the transcripts.
import json


class DBConnector:
    """
    The class to hold the methods to connect to the DB to pull the rules dictionary, labels dictionary and tags added
    transcripts. Push the corrected transcripts back to the DB.
    """

    def __init__(self, path_transcripts=None, path_rules=None, path_labels=None):

        self.path_transcripts = path_transcripts
        self.path_rules = path_rules
        self.path_labels = path_labels

    def load_transcripts(self):
        """
        Current version supports local File system loading of the transcripts.
        Further scope is to add the DB(DocumentDB) connection here.
        """
        with open(self.path_transcripts) as file:
            transcripts = json.load(file)

        return transcripts

    def load_rule_dict(self):
        """
        Current version supports local File system loading of the transcripts and all the below operations.
        Further scope is to add the DB(DocumentDB) connection here.
        """
        with open(self.path_rules) as file:
            rules_dict = json.load(file)

        return rules_dict

    def load_label_dict(self):
        with open(self.path_labels) as file:
            labels_dict = json.load(file)

        return labels_dict

    def push_to_db(self, store_loc, clean_transcript):

        with open(store_loc, "w") as file:
            json.dump(clean_transcript, file)
