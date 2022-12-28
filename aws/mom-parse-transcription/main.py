import json
import logging
import os

import boto3
import constants
from dotenv import dotenv_values
from utils import (S3Adapter, TranscribeUtils, create_s2t_response,
                   create_s2t_success_message, file_status_update,
                   get_base_path, get_bucket_name, get_diarization_extension,
                   get_store_path, parse_media_uri, parse_transcript_uri)

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_transcription_handler(event, context):

    # Creates an OrderedDict of env variables from .env file
    soa_config = dict(dotenv_values(".env"))

    # Initializes a transcribe client for interacting with AWS transcribe
    transcribe_client = boto3.client("transcribe")

    # Extracts job details(jon_name and job_status) from lambda event
    logger.info(f"Extracting job_name and status from lambda event")
    job_name = event.get("detail").get("TranscriptionJobName")
    job_status = event.get("detail").get("TranscriptionJobStatus")
    # Adding the log statements for further monitoring
    logger.info("Extraction of job_name and response has been successfully completed")
    logger.info(f"Job name --> {job_name}  Job status --> {job_status}")

    # Get's job_response from corresponding transcription job
    logger.info(f"Generating transcription job response from --> {job_name}")
    job_response = transcribe_client.get_transcription_job(
        TranscriptionJobName=job_name
    )
    logger.info(f"Transcription job response generated successfully --> {job_response}")

    # Get's diarization/extension value using utils.get_diarization_extension function
    diarization, extension = get_diarization_extension(
        transcribe_job_response=job_response
    )

    # Get media URI from corresponding transcription job response
    media_uri = job_response.get("TranscriptionJob").get("Media").get("MediaFileUri")
    # Let's add some log statements for further monitoring
    logger.info(f"Speech to Text Input file path --> {media_uri}")

    # Extracts bucket name and base/file name from MediaURI
    file_name = os.path.basename(media_uri)
    bucket_name = get_bucket_name(s3_uri=media_uri)
    logger.info(f"Bucket Name for retrieving Transcribe response --> {bucket_name}")

    # Initializes S3Utils object for utility functions
    s3_utils = S3Adapter(bucket_name=bucket_name)

    # To get base path for stroing speech to text output
    base_path = get_base_path(path=media_uri)
    logger.info(f"Base Path for retrieving Transcribe response --> {base_path}")

    # Logic to push response path and Speaker Diarization? value into SNS topic
    if job_status == constants.COMPLETED:
        logger.info(f"Transcription Job - {job_name} got {constants.COMPLETED}")

        # Get speech to text output from transcription job response
        s2t_output_uri = (
            job_response.get("TranscriptionJob")
            .get("Transcript")
            .get("TranscriptFileUri")
        )

        # Parses media uri and transcript uri by calling parse utility functions
        media_path = f"{constants.S3_URI}{parse_media_uri(file_uri=media_uri)}"
        s2t_output_path = (
            f"{constants.S3_URI}{parse_transcript_uri(file_uri=s2t_output_uri)}"
        )
        # Log statements for further monitoring
        logger.info(f"Speech to Text Input file path --> {media_path}")
        logger.info(f"Speech to Text Output file path --> {s2t_output_path}")

        prefix_exist, file_prefix = s3_utils.prefix_exist(file_prefix=s2t_output_path)
        if not prefix_exist:
            logger.info(
                f"{file_prefix} doesn't exist so skipping initialization of transcription job"
            )

            # Initializes a transcribe utility object
            transcribe_utils = TranscribeUtils(
                s3_utils=s3_utils, job_name=job_name, job_status=job_status
            )

            # Handles the failed transcription job by deleting it
            error_message = transcribe_utils.handle_transcribe_failure(
                extension=extension,
                file_name=file_name,
                base_path=base_path,
                media_path=media_uri,
                diarization=diarization,
            )

            # Calls file_status_update function to update status of api_file table
            status = file_status_update(
                status=constants.ERROR_STATUS, file_name=media_uri
            )
        else:
            # Checks whether diarization is enabled or not
            if diarization:
                logger.info("Diarization is enabled on aws transcribe job")

                # Creates a message dict by calling create_s2t_success_message function
                s2t_success_message = create_s2t_success_message(
                    extension=extension,
                    media_path=media_path,
                    diarization=diarization,
                    status=constants.SUCCESS,
                    s2t_output_path=s2t_output_path,
                )

                # Let's add some log statements for further monitoring
                logger.info(
                    "Started publishing Speech to Text message into {diarization} topic"
                )
                # Publisges a message using publish_message function
                response = s3_utils.publish_message(
                    message=s2t_success_message,
                    subject=constants.S2T_SUBJECT,
                    topic_arn=soa_config.get("DIARIZATION_TOPIC_ARN"),
                )

                # Log statements for further monitoring
                logger.info(
                    "Speech to Text message published successfully into the {diarization} topic"
                )
                logger.info(
                    f"Speech to Text success message confirmation  --> {response}"
                )

            else:
                logger.info("Diarization is not enabled on aws transcribe job")

                # Get store path for stroing speech to  response
                store_path = get_store_path(base_path=base_path)

                # Extracts transcript from aws transcribe output
                logger.info(f"Reading AWS Transcribe output from {s2t_output_path}")
                transcript = (
                    s3_utils.read_json(read_path=s2t_output_path)
                    .get(constants.RESULT)
                    .get(constants.TRANSCRIPTS)[0]
                )
                logger.info(f"AWS Transcribe Output {transcript}")

                # Creates speech to text response template
                logger.info(
                    "Creating Speech to text response from AWS Transcribe response"
                )
                s2t_response = create_s2t_response(
                    status=constants.SUCCESS, transcript=transcript
                )
                logger.info(
                    f"Speech to Text response created successfully --> {s2t_response}"
                )

                # Put the speech to text response in s2t output path
                logger.info(f"Storing Speech to Text output in {s2t_output_path}")
                s3_utils.put_data(
                    store_path=s2t_output_path, content=json.dumps(s2t_response)
                )
                logger.info(
                    f"Speech to text output saved successfully in {s2t_output_path}"
                )

                # Creates a message dict by calling create_s2t_success_message function
                s2t_success_message = create_s2t_success_message(
                    status=constants.SUCCESS,
                    diarization=diarization,
                    extension=extension,
                    media_path=media_path,
                    s2t_output_path=s2t_output_path,
                )

                # Publisges a message using publish_message function
                response = s3_utils.publish_message(
                    message=s2t_success_message,
                    subject=constants.S2T_SUBJECT,
                    topic_arn=soa_config.get("DOWNSTREAM_TOPIC_ARN"),
                )

                logger.info(f"Speech to Text message confirmation  --> {response}")

    elif job_status == constants.FAILED:
        logger.info(f"Transcription Job - {job_name} got {constants.FAILED}")

        # Initializes a transcribe utility object
        transcribe_utils = TranscribeUtils(
            s3_utils=s3_utils, job_name=job_name, job_status=job_status
        )

        # Handles the failed transcription job by deleting it
        error_message = transcribe_utils.handle_transcribe_failure(
            extension=extension,
            diarization=diarization,
            file_name=file_name,
            base_path=base_path,
            media_path=media_uri,
        )

        # Calls file_status_update function to update status of api_file table
        status = file_status_update(status=constants.ERROR_STATUS, file_name=media_uri)
