import logging
import re

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class TextProcessing:
    """
    This class is responsible for processing the text like,
    removing extra spaces, punctuations etc.
    """

    @staticmethod
    def remove_extra_spaces(transcript):
        """
        To remove extra spaces from text sentence
        """

        # Logging for further monitoring
        logger.info("Removing extra spaces from segmented transcript")
        return re.sub(r"\s+", " ", transcript).strip()

    @staticmethod
    def remove_punctuation_spaces(transcript):

        # Logging for further monitoring
        logger.info("Removing punctuations from transcript")
        return re.sub(r'\s([?.!,"](?:\s|$))', r"\1", transcript)

    @staticmethod
    def parse_punctuations(transcript):

        # Logging for further monitoring
        logger.info("Removing unnecessary punctuations from transcript")
        transcript = re.sub(r"\s*[,;:\-\"\']\s*$", "", transcript)
        if not transcript.endswith("."):
            transcript += "."
        return transcript

    @staticmethod
    def capitalize(transcript):

        # Logging for further monitoring
        logger.info("Capitalizing the transcript")
        return transcript[0].upper() + transcript[1:]

    @staticmethod
    def process_transcript(transcript):
        """
        This function is the entrypoint function for text processing
        """

        # Calling all the static methods in TextProcessing class
        transcript = TextProcessing.capitalize(
            transcript=TextProcessing.parse_punctuations(
                transcript=TextProcessing.remove_punctuation_spaces(
                    transcript=TextProcessing.remove_extra_spaces(transcript=transcript)
                )
            )
        )
        logger.info("Text processing has been completed")
        return transcript
