import json
import logging
import os
import sys

import config
import constants
from dotenv import dotenv_values
from engine import DiarizeEngine
from utils import (S3Adapter, create_response, create_success_message,
                   file_status_update, get_bucket_name, get_segments_path,
                   get_store_path, handle_failure)

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def diarization_handler(event, context):
    """
    Entrypoint function for diarization wrapper which diarizes
    the transcription output and once the job is completed it stores
    the diarization output into S3 bucket.

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

    # Get message which is sent to Diarization SNS Topic by parse_transcription
    message = json.loads(event["Records"][0]["Sns"]["Message"])
    logger.info("Message received successfully from diarization-topic")
    logger.info(f"Speech To Text message --> {message}")

    # Extracts media_path and s2t_output_path from SNS message
    media_path = message.get("media_path")
    s2t_output_path = message.get("s2t_output_path")
    # Let's logs input file path to aws transcribe and speech to text output path
    logger.info(f"Diarization Input file path --> {media_path}")
    logger.info(f"Speech to text output path --> {s2t_output_path}")

    # Extracts diarization and media_path extesion from SNS message
    extension = message.get("extension")
    logger.info(f"Diarization Input Media Type --> {extension}")
    diarization = message.get("diarization")

    # Get bucket_name for diarization store the output
    bucket_name = get_bucket_name(s3_uri=media_path)
    logger.info(f"Bucket name for storing diarization response --> {bucket_name}")
    # Initializes s3_utils object using which we can access s3 utility functions
    s3_utils = S3Adapter(bucket_name=bucket_name)

    # Get store path for diarization to store the output Json
    store_path = get_store_path(file_uri=media_path, bucket_name=bucket_name)
    logger.info(f"Diarization Output path --> {store_path}")

    # Get speech to text success, if status is error function exits
    s2t_status = message.get(constants.STATUS)

    if s2t_status == constants.ERROR:
        logger.info(
            "Speech to text/AWS Transcribe service failed, so moving diarization also to fail state"
        )

        # Get downstream topic arn from .env file to be used to publish SNS message
        downstream_topic_arn = soa_config.get("DOWNSTREAM_TOPIC_ARN")

        # Creates diarization error response/message and publishes, stores into SNS/S3
        error_response, error_message = handle_failure(
            extension=extension,
            media_path=media_path,
            diarization=diarization,
            s2t_output_path=s2t_output_path,
            diarization_output_path=store_path,
            downstream_topic_arn=downstream_topic_arn,
        )

        # Saving the diarization error response into store_path of diarization
        logger.info(f"Storing Diarization error response into --> {store_path}")
        s3_utils.put_data(store_path=store_path, content=json.dumps(error_response))
        logger.info(f"Diarization error response saved successfully into {store_path}")

        # Saving the diarization error response into store_path of speech to text
        logger.info(f"Storing Speech to Text error response into --> {s2t_output_path}")
        s3_utils.put_data(store_path=s2t_output_path, content=json.dumps(s2t_response))
        logger.info(
            f"Speech to Text error response saved successfully into {s2t_output_path}"
        )

        # Publishes error message into Downstream SNS Topic
        response = s3_utils.publish_message(
            subject=constants.ERROR_SUBJECT,
            diarization_message=error_message,
            topic_arn=soa_config.get("DOWNSTREAM_TOPIC_ARN"),
        )
        logger.info(f"Diarization error message confirmation  --> {response}")

        # Calls file_status_update function to update status of api_file table
        file_status_update(status=constants.ERROR_STATUS, file_name=media_path)

    else:

        # To get the diarization/speech to text constants
        diarization_specs = constants.DIARIZATION
        s2t_specs = constants.SPEECH_TO_TEXT

        # Checks whether diarization is enabled on aws transcribe if not it returns transcript
        if not int(diarization):
            s2t_output = {}

            # Let's log some information for further monitoring
            logger.info(
                f"Diarization is disabled so skipping execution of diarization engine"
            )

            # Extracts transcript from aws transcribe output
            transcript = (
                s3_utils.read_json(read_path=s2t_output_path)
                .get(constants.RESULT)
                .get(constants.TRANSCRIPTS)[0]
            )
            logger.info(f"Storing Speech to Text output in {s2t_output_path}")

            # Put the speech to text response in s2t output path
            s3_utils.put_data(
                store_path=s2t_output_path, content=json.dumps(transcript)
            )
            logger.info(
                f"Speech to text output saved successfully in {s2t_output_path}"
            )

        else:

            # Get video/audio store path to store video/audio segments
            audio_path, video_path = get_segments_path(
                file_uri=media_path, bucket_name=bucket_name
            )

            # Updates env dict with bucket_name, audio/video store paths
            soa_config.update(
                {
                    **soa_config,
                    **{
                        "AUDIO_PATH": audio_path,
                        "VIDEO_PATH": video_path,
                        "BUCKET_NAME": bucket_name,
                        "MEDIA_PATH": media_path,
                        "S2T_OUTPUT_PATH": s2t_output_path,
                    },
                }
            )

            # Initializes the dataclass object from env variables
            settings = config.Settings.from_dict(soa_config)

            # Initializes a DiarizationEngine object to diarize
            model = DiarizeEngine(settings=settings, extension=extension)

            # Call the diarize function using model object created from DiarizeEngine class
            diarization_success, response = model.diarize()

            if diarization_success:
                # Log statements for further monitoring
                logger.info("Diarization process successfully completed")
                logger.info(f"Diarization response - {response}")

                # Let's create a response Json for MOM
                diarization_response = create_response(
                    response=response,
                    status=constants.SUCCESS,
                    model_name=diarization_specs.get("model_val"),
                )
                # Saving the diarization response into store_path of diarization
                logger.info(
                    f"Storing Diarization success response into --> {store_path}"
                )
                s3_utils.put_data(
                    store_path=store_path,
                    content=json.dumps(diarization_response),
                )

                logger.info(
                    f"Diarization success response saved successfully into {store_path}"
                )

                # Saving the diarization response into store_path of speech to text
                logger.info(
                    f"Storing Speech to Text success response into --> {s2t_output_path}"
                )
                s2t_response = create_response(
                    response=response,
                    status=constants.SUCCESS,
                    model_name=s2t_specs.get("model_val"),
                )
                s3_utils.put_data(
                    store_path=s2t_output_path,
                    content=json.dumps(s2t_response),
                )
                logger.info(
                    f"Speech to Text success response saved successfully into {s2t_output_path}"
                )

                # Push the message into SNS downstream topic
                success_message = create_success_message(
                    extension=extension,
                    media_path=media_path,
                    diarization=diarization,
                    s2t_output_path=s2t_output_path,
                    diarization_output_path=store_path,
                )

                response = s3_utils.publish_message(
                    diarization_message=success_message,
                    subject=constants.DIARIZATION_SUBJECT,
                    topic_arn=soa_config.get("DOWNSTREAM_TOPIC_ARN"),
                )
                logger.info(f"Diarization success message confirmation  --> {response}")

            else:
                # Log statements for further monitoring
                logger.info("Diarization process has been failed")

                # Get downstream topic arn from .env file to be used to publish SNS message
                downstream_topic_arn = soa_config.get("DOWNSTREAM_TOPIC_ARN")

                # Creates diarization error response/message and publishes, stores into SNS/S3
                (
                    error_message,
                    s2t_error_response,
                    diarization_error_response,
                ) = handle_failure(
                    extension=extension,
                    media_path=media_path,
                    diarization=diarization,
                    s2t_output_path=s2t_output_path,
                    diarization_output_path=store_path,
                    downstream_topic_arn=downstream_topic_arn,
                )

                # Saving the S2T error response into store_path of speech to text
                logger.info(
                    f"Storing Speech to Text error response into --> {s2t_output_path}"
                )
                s3_utils.put_data(
                    store_path=s2t_output_path,
                    content=json.dumps(s2t_error_response),
                )
                logger.info(
                    f"Speech to Text error response saved successfully into {s2t_output_path}"
                )

                # Saving the diarization error response into store_path of diarization
                logger.info(f"Storing Diarization error response into --> {store_path}")
                s3_utils.put_data(
                    store_path=store_path,
                    content=json.dumps(diarization_error_response),
                )
                logger.info(
                    f"Diarization error response saved successfully into {store_path}"
                )

                # Publishes error message into Downstream SNS Topic
                response = s3_utils.publish_message(
                    subject=constants.ERROR_SUBJECT,
                    diarization_message=error_message,
                    topic_arn=soa_config.get("DOWNSTREAM_TOPIC_ARN"),
                )
                logger.info(f"Diarization error message confirmation  --> {response}")

                # Calls file_status_update function to update status of api_file table
                file_status_update(status=constants.ERROR_STATUS, file_name=media_path)
