""" [AWS Audio Transcribe Module]

This module initiates/returns the transcription job id
of the media file stored on S3.

    """

import json
import logging
import os

import aws_transcribe
import config
import constants
from dotenv import dotenv_values
from utils import (S3Adapter, create_error_message, do_diarization,
                   file_status_update, get_bucket_name,
                   get_diarization_store_path, get_store_path, handle_failure)

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def transcription_handler(event, context):
    """
    Entrypoint function for generic transcribe wrapper which initiates
    the transcription job and once the job is completed it stores
    the output into S3 bucket.

    Parameters
    ----------
        :param event (dict): Lambda event dictionary.
        :param context(LambdaContext()): Lambda Context

    Returns
    -------
        (dict): transcription_job_name with relevant headers
         in form of dictionary
    """
    # Creates an OrderedDict of env variables from .env file
    soa_config = dict(dotenv_values(".env"))

    # Parsing query parameters from event
    query_params = event.get("queryStringParameters")
    if not query_params:
        query_params = {}

    # Getting file_uri/input to aws transcribe
    file_uri = query_params.get("file_uri")

    # Publishes message into Downstream SNS topic if file_uri param is None
    if not file_uri:
        # Initializes s3_utils object using which we can access s3 utility functions
        s3_utils = S3Adapter()

        logger.info("No file specified for transcribing")

        # Creates error response for publishing into Downstream SNS Topic
        error_response = create_error_message(media_path=file_uri)

        # Publisges a message using publish_message function
        response = s3_utils.publish_message(
            message=error_response,
            topic_arn=soa_config.get("DOWNSTREAM_TOPIC_ARN"),
            subject=constants.ERROR_SUBJECT,
        )
        logger.info(f"Error message confirmation  --> {response}")

        return {
            "statusCode": 400,
            "status": constants.ERROR,
            "isBase64Encoded": False,
            "body": "No file specified for transcribing so skipping initialization of transcription job",
        }

    else:
        # Get bucket name from file_uri paramter
        bucket_name = get_bucket_name(s3_uri=file_uri)

        # Initializes s3_utils object using which we can access s3 utility functions
        s3_utils = S3Adapter(bucket_name=bucket_name)

        logger.info(f"Input media file URI is {file_uri}")
        prefix_exist, file_prefix = s3_utils.prefix_exist(file_prefix=file_uri)
        if not prefix_exist:
            logger.info(
                f"{file_prefix} doesn't exist so skipping initialization of transcription job"
            )

            # Creates error response for publishing into Downstream SNS Topic
            error_response = create_error_message(media_path=file_uri)

            # Publisges a message using publish_message function
            response = s3_utils.publish_message(
                message=error_response,
                topic_arn=soa_config.get("DOWNSTREAM_TOPIC_ARN"),
                subject=constants.ERROR_SUBJECT,
            )
            logger.info(f"Error message confirmation  --> {response}")

            return {
                "statusCode": 400,
                "status": constants.ERROR,
                "isBase64Encoded": False,
                "body": f"{file_prefix} doesn't exist so skipping initialization of transcription job",
            }

        # Getting store path for aws transcribe
        store_path = query_params.get("store_prefix")
        store_path = get_store_path(
            file_uri=file_uri, store_path=store_path, bucket_name=bucket_name
        )
        logger.info(f"AWS Transcribe/Speech To Text Output Location --> {store_path}")

        # do_diarization function decides whether or not diarization to be performed
        diarization = do_diarization(
            config=soa_config, diarization=query_params.get("diarization")
        )

        # Updating env dict with file/S3 URI
        soa_config.update(
            {**soa_config, **{"FILE_URI": file_uri, "SPEAKER_DIARIZATION": diarization}}
        )
        # Initializes the dataclass object from env variables
        settings = config.Settings.from_dict(soa_config)

        # Initiates the aws transcribe transcription job
        logger.info("Transcribe process has been started")
        transcribe = aws_transcribe.Transcribe(settings=settings)
        transcribe_success, transcribe_job_name = transcribe.transcribe_file(
            store_path=s3_utils.get_prefix_from_uri(path=store_path)
        )

        # Checks whether transcription process is successfull or not
        if transcribe_success:
            logger.info("Transcribe process has been completed")

            # Calls file_status_update function to update status of api_file table
            status = file_status_update(
                status=constants.PROCESSING_STATUS, file_name=file_uri
            )

            # Returns response dictionary to the caller function
            return {
                "statusCode": 200,
                "status": constants.PROCESSING_STATUS,
                "isBase64Encoded": False,
                "body": transcribe_job_name,
            }

        else:
            logger.info("Transcribe process has been failed")

            # Get Diarization store path to store diarization error response
            diarization_store_path = get_diarization_store_path(
                file_uri=file_uri, bucket_name=bucket_name
            )

            # Creates error response for publishing into Downstream SNS Topic
            (
                error_message,
                s2t_error_response,
                diarization_error_response,
            ) = handle_failure(
                media_path=file_uri,
                diarization=diarization,
                s2t_store_path=store_path,
                extension=settings.EXTENSION,
                diarization_store_path=diarization_store_path,
            )

            # Saving the S2T error response into store_path of speech to text
            logger.info(f"Storing Speech to Text error response into --> {store_path}")
            s3_utils.put_data(
                store_path=store_path,
                content=json.dumps(s2t_error_response),
            )
            logger.info(
                f"Speech to Text error response saved successfully into {store_path}"
            )

            # Saving the diarization error response into store_path of diarization
            logger.info(
                f"Storing Diarization error response into --> {diarization_store_path}"
            )
            s3_utils.put_data(
                store_path=diarization_store_path,
                content=json.dumps(diarization_error_response),
            )
            logger.info(
                f"Diarization error response saved successfully into {diarization_store_path}"
            )

            # Publisges a message using publish_message function
            response = s3_utils.publish_message(
                message=error_message,
                subject=constants.ERROR_SUBJECT,
                topic_arn=settings.DOWNSTREAM_TOPIC_ARN,
            )
            logger.info(f"Error message confirmation  --> {response}")

            # Calls file_status_update function to update status of api_file table
            status = file_status_update(
                status=constants.ERROR_STATUS, file_name=file_uri
            )

            # Returns response dictionary to the caller function
            return {
                "statusCode": 400,
                "status": constants.ERROR_STATUS,
                "isBase64Encoded": False,
                "body": transcribe_job_name,
            }
