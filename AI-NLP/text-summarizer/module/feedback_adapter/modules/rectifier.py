import logging
import re
import sys

from module.feedback_adapter import config

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


class TransRectifier:
    def __init__(self, transcripts=None, rules=None):
        """
        This class should hold methods to rectify different transcript using rule pattterns.

        """
        self.transcripts = transcripts
        self.dict_rules = rules

    def rectify(self):
        # Append it to the clean transcripts.
        # This method uses the self.transcripts and the self.dict_rules.
        for transcript_dict in self.transcripts:
            clean_transcript = self.cleanup(
                transcript_dict["transcript"], self.dict_rules
            )
            transcript_dict["transcript"] = clean_transcript
        return self.transcripts

    @staticmethod
    def cleanup(transcript, rules_dict):
        """
        This static method ingests a transcript and then find the pattern of the transcription rules in the same, and
        then applies the rule for the cleanup.
        """
        # [SOT] _ [EOT] text encapsulation, target text for rectification.
        if transcript.find("[SOT]") != -1:
            start = transcript.find("[SOT]") + len("[SOT]")
            end = transcript.find("[EOT]")
            transcript = transcript[start:end]

            # Perform the rules cleanup action
            transcript = apply_rules(transcript, rules_dict)

        return transcript


# Rules Application Utilities
def apply_rules(transcript, rules_dict):
    """

    Args:
        transcript: The target rectification string object
        rules_dict: The rule patterns to be used for the cleanup

    Returns: Cleaned up transcript -> string object

    """

    # Check what rules are getting applied in the transcript and then perform the cleanup action based on those rules
    # only. Find the rules that are needed to be used for a particular transcript.

    applied_rules = []

    for rule in rules_dict.keys():
        # Split the tags for a rule into list.
        list_tags = rules_dict[rule].split(" ")

        # Create a list which is used to validate whether the rule matches or not.
        # Checking for at-least one instance of the rule.
        rule_validator = []
        for tag in list_tags:
            if tag in transcript:
                rule_validator.append(1)
            else:
                rule_validator.append(0)

        # The rule_validator list sum [1, ... ,1] matches the length of the tags.
        if sum(rule_validator) == len(list_tags):
            applied_rules.append(rule)

    # Push the speaker separation rule to the end of the applied rules list.
    for i in applied_rules:
        if "[sp1]" == rules_dict[i]:
            applied_rules.remove(i)
            applied_rules.append(
                list(rules_dict.keys())[list(rules_dict.values()).index("[sp1]")]
            )
            break

    modified_transcript = transcript
    for rule in applied_rules:
        # Invoke the rules pattern matching function to apply the rules.
        modified_transcript = getattr(sys.modules[__name__], "%s" % rule.lower())(
            modified_transcript
        )

    return modified_transcript


# Below are rules pattern matching functions
def r1(transcript):
    """
    rule : Extra words by machine removed after applying this rule.

    Args:
        transcript: Rules pattern added string object.

    Returns: String object with all the extra words removed.

    """

    pattern = "\[\](.*?)\[[/]\]"
    res = re.sub(pattern, "", transcript)

    return res


def r2(transcript):
    """
    rule : Words missed by the machine.

    Args:
        transcript: Rules pattern added string object.

    Returns: String object with all the missing words added.

    """
    pattern1 = "\[\-\]"
    pattern2 = "\[[/]\-\]"
    res = re.sub(pattern1, "", transcript)
    res = re.sub(pattern2, "", res)
    return res


def r3(transcript):
    """
    rule : Speaker Separation

    Args:
        transcript: The rules added transcript strings for the rectification.

    Returns: dictionary, where each element is speakers separated.
    Multiple speaker parsing - dict sequenced with speaker number, 0, 1, 2, 3, etc.
    {"spk1": [ utterance 1, utterance 2], "spk2": [utterance 1], "spk3": [utterance1, utterance2]}

    """
    dict_spk = {}
    for i in range(1, config.NUM_SPEAKER + 1):
        pattern = f"\[(sp{i})\](.*?)\[(sp{i})\]"
        if bool(re.findall(pattern, transcript)):
            res = re.findall(pattern, transcript)
            dict_spk[f"sp{i}"] = []
            for spk_inst in res:
                dict_spk[f"sp{i}"].append(spk_inst[1])

    return dict_spk


def r4(transcript):
    """
    rule : Machine totally skipped to record a sentence associated with audio chunks.

    Args:
        transcript: Rules pattern added string object.

    Returns: String object with added missing sentence in the transcript.

    """
    pattern1 = "\[\$\]"
    res = re.sub(pattern1, "", transcript)
    pattern2 = "\[[/]\$]"
    res = re.sub(pattern2, "", res)
    return res


def r5(transcript):
    """
    rule : misinterpreted word

    Args:
        transcript: Rules pattern added string object.

    Returns: String object with word replaced with correct word.

    """
    pattern1 = "\<\>(.*?)\[\#\]"
    res = re.sub(pattern1, "", transcript)
    pattern2 = "\[[/]\#\]\<[/]\>"
    res = re.sub(pattern2, "", res)

    return res


def r6(transcript):
    """
    rule : misinterpreted sentence.

    Args:
        transcript: Rules pattern added string object.

    Returns: String object with replaced sentence words in the transcript.

    """
    pattern1 = "\(\)(.*?)\[\#\]"
    res = re.sub(pattern1, "", transcript)
    pattern2 = "\[[/]\#\]\([/]\)"
    res = re.sub(pattern2, "", res)

    return res


def r7(transcript):
    """
    rule : alphabets to numerical mismatch.

    Args:
        transcript: Rules pattern added string object.

    Returns: The words converted to numeric format where ever required, return the transcript.

    """
    pattern1 = "\[\@\](.*?)\[\#\]"
    res = re.sub(pattern1, "", transcript)
    pattern2 = "\[[/]\#\]\[[/]\@\]"
    res = re.sub(pattern2, "", res)

    return res


def r8(transcript):
    """
    rule : word recorded wrong w.r.t its spelling.

    Args:
        transcript: Rules pattern added string object.

    Returns: correct the words such as names wrongly spelt in the transcript and return the string object.

    """
    pattern1 = "\[(sc)](.*?)\[\#\]"
    res = re.sub(pattern1, "", transcript)
    pattern2 = "\[[/]\#\]\[[/](sc)]"
    res = re.sub(pattern2, "", res)

    return res


def rectifier_validator(gt, generated):

    for transcript_gt, transcript_pred in zip(gt, generated):
        logging.debug("")
        logging.debug(f"GT : {transcript_gt}")
        logging.debug(f"Clean Generated : {transcript_pred}")

        if type(transcript_gt) is str:
            if transcript_gt.replace(" ", "") == transcript_pred.replace(" ", ""):
                logging.debug("Test Case - Passed")
            else:
                logging.debug("Test Case - Failed")

        # Check the speaker separation output validation.
        else:

            # Check if the number of speakers are correct
            if transcript_gt.keys() == transcript_pred.keys():
                spkr_valid_rect = []
                for spk in transcript_gt.keys():
                    # List to valid all the utterances per speaker list
                    utter_valid_list = []
                    for utterance_gt, utterance_pred in zip(
                        transcript_gt[spk], transcript_pred[spk]
                    ):
                        if utterance_gt.replace(" ", "") == utterance_pred.replace(
                            " ", ""
                        ):
                            utter_valid_list.append(1)
                        else:
                            utter_valid_list.append(0)

                    # If the total 1 in the speaker valid list matches the total number of utterances of a speaker
                    # Then the speaker utterance test is passed.
                    if sum(utter_valid_list) == len(transcript_gt[spk]):
                        spkr_valid_rect.append(1)
                    else:
                        spkr_valid_rect.append(0)

                # If the total 1s in the transcription validation list matches the number of
                # speaker in the transcription.
                # All the speakers are validated then.

                if sum(spkr_valid_rect) == len(transcript_gt.keys()):
                    logging.debug("Test case - Passed")
                else:
                    logging.debug("Test Case - Failed")

            else:
                logging.debug("Test Case - Failed : Number of speakers doesnot match!!")
