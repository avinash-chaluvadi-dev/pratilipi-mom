""" [AWS Transcribe Module]

This module holds all the classes required for initiating 
and processing Transcription Jobs(Speech-to-text)

This file can also be imported as a module and contains the following
classes:

    * Transcribe - Base custom class for transcribe
    
"""

import logging
import time

import boto3
import constants
from botocore.exceptions import ClientError, ParamValidationError
from utils import (S3Adapter, get_bucket_name, get_transcribe_job_name,
                   timeit_logger)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class Transcribe:
    """
    This class initiates the transcription job and processes it.

    Methods
    -------
    __init__()
        Initializes the state of object with the input keyword arguments

    transcribe_file()
        Transcribes the media file present in S3

    """

    def __init__(self, settings):
        """
        Constructor of Transcribe class to initialize the state of object

        Parameters
        ----------
            :param settings (dict): settings of audio-transcribe
        """
        self.settings = settings
        self.transcribe_client = boto3.client("transcribe")

    @timeit_logger
    def transcribe_file(self, store_path: str):
        """
        Initiates the transcribe job in aws transcribe, the audio file must
        already be present in S3 bucket specified in the environment variables,
        if the file is not present the function execution will be skipped.

        Parameters
        ----------
            :param store_path (str): start_transcription_job() func uses store_path to store output

            :raises RuntimeError which should caught at main level and returned a proper response to transcribe

        """
        # Parses/Gets bucket name from S3 URI/constants module
        bucket_name = get_bucket_name(s3_uri=self.settings.FILE_URI)

        # Gets a unique name for aws transcribe to use while creating a
        transcribe_job_name = get_transcribe_job_name()

        try:
            # Initializes speaker_identification_settings based on SPEAKER_DIARIZATION env variable
            if self.settings.SPEAKER_DIARIZATION:
                speaker_identification_settings = {
                    "ShowSpeakerLabels": True,
                    "MaxSpeakerLabels": 10,
                }
            else:
                speaker_identification_settings = {}

            # Initializes/Starts transcription job using start_transcription_job func
            response = self.transcribe_client.start_transcription_job(
                TranscriptionJobName=transcribe_job_name,
                MediaFormat=self.settings.EXTENSION,
                Media={"MediaFileUri": self.settings.FILE_URI},
                OutputBucketName=bucket_name,
                OutputKey=store_path,
                IdentifyLanguage=True,
                Settings=speaker_identification_settings,
            )
            logger.info(f"{transcribe_job_name} job has been created")
            return constants.TRANSCRIBE_SUCCESS, transcribe_job_name

        except (
            RuntimeError,
            ClientError,
            ParamValidationError,
            ValueError,
            TypeError,
        ) as error:

            # Add some log statements for further monitoring
            logger.error("Exception occurred while initializing a transcription job")

            return constants.TRANSCRIBE_FAIL, transcribe_job_name
