import datetime
import logging
import shlex
import subprocess
from string import punctuation

import boto3
import constants
from utils import (S3Adapter, clear_directory, get_segment_transcript,
                   get_segments_chunk, get_temp_store_path,
                   initilaize_segment_dict)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class DiarizeEngine:
    def __init__(self, settings, extension):
        self.settings = settings
        self.extension = extension
        self.s3_client = boto3.client("s3")
        self.s3_utils = S3Adapter(bucket_name=self.settings.BUCKET_NAME)

    def trim_audio(self, start_time, end_time, media_presigned_uri, audio_path):
        """This function trims audio files from media file based on start and end_time"""

        # Let's add some log statements for further monitoring
        logger.info(
            f"Audio segments will be saved into audio segment prefix --> {self.settings.AUDIO_PATH}"
        )
        logger.info(f"Segmenting audio file from {start_time} to {end_time}")

        # Generates ffmpeg cli command to trim audio files
        ffmpeg_cmd = f"/opt/ffmpeglib/ffmpeg -i {media_presigned_uri} -ss {start_time} -to {end_time} {constants.AUDIO_COMMAND_SUFFIX}"
        # ffmpeg_cmd = f"/opt/ffmpeglib/ffmpeg -i {media_presigned_uri} -ss {start_time} -to {end_time} -f mpegts -q:a 0 -map a -"

        # Splits the ffmpeg command into a list of inputs and sub commands
        trim_command = shlex.split(ffmpeg_cmd)

        # Call the ffmpeg command line utility func using subprocess
        process_object = subprocess.run(
            trim_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Writes the audio bytes into S3 Bucket using S3Adapter utility class
        self.s3_utils.put_data(
            store_path=audio_path,
            content=process_object.stdout,
            content_type=constants.AUDIO_CONTENT_TYPE,
        )

        logger.info("Audio file segmented successfully")
        logger.info(f"Segmented audio file saved into {audio_path}")

    def trim_video(self, start_time, end_time, media_presigned_uri, video_path):
        """This function trims video files from media file based on start and end_time"""

        # Let's add some log statements for further monitoring
        logger.info(
            f"Video segments will be saved into audio segment prefix --> {self.settings.VIDEO_PATH}"
        )
        logger.info(f"Segmenting video file from {start_time} to {end_time}")

        # Get temp store path to store video files temprarily
        temp_store_path = get_temp_store_path(file_name=video_path)

        # Generates ffmpeg cli command to trim audio files
        ffmpeg_cmd = f"/opt/ffmpeglib/ffmpeg -i {media_presigned_uri} -ss {start_time} -to {end_time} {constants.VIDEO_COMMAND_SUFFIX} {temp_store_path}"
        # ffmpeg_cmd = f"/opt/ffmpeglib/ffmpeg -i {media_presigned_uri} -ss {start_time} -to {end_time} -f mp4 -c:v copy -c:a copy -"

        # Splits the ffmpeg command into a list of inputs and sub commands
        trim_command = shlex.split(ffmpeg_cmd)

        # Call the ffmpeg command line utility func using subprocess
        process_object = subprocess.run(
            trim_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Uploads the video bytes into S3 Bucket using S3Adapter utility class
        self.s3_utils.upload_file(
            temp_store_path=temp_store_path, s3_store_path=video_path
        )

        # Clears /tmp folder by removing the ffmpeg stored video file in every iteration
        clear_directory(path=temp_store_path)

        logger.info("Video file segmented successfully")
        logger.info(f"Segmented audio file saved into {video_path}")

    def trim(self, start_time, end_time):
        """
        This function trims the audio and video files
        based on start and end time
        """

        # Get audio/video basenames which are to be used further to store segments
        audio_path, video_path = get_segments_chunk(
            start_time=start_time,
            end_time=end_time,
            audio_path=self.settings.AUDIO_PATH,
            video_path=self.settings.VIDEO_PATH,
            extension=self.extension,
        )

        # Generates time in HH:MM:SS format
        start_time = str(datetime.timedelta(seconds=start_time))
        end_time = str(datetime.timedelta(seconds=end_time))

        # Generates pre-signed uri of media file using generate_presigned_uri func
        media_presigned_uri = self.s3_utils.generate_presigned_uri(
            path=self.settings.MEDIA_PATH,
            uri_timeout=self.settings.SIGNED_URL_TIMEOUT,
        )
        logger.info(f"PRESIGNED URI --> {media_presigned_uri}")

        # Call the trim_audio func to trim audio from MEDIA_PATH
        self.trim_audio(
            start_time=start_time,
            end_time=end_time,
            media_presigned_uri=media_presigned_uri,
            audio_path=audio_path,
        )

        # Checks the extension type and calls trim_video if extension type is video
        if self.extension in constants.VIDEO_EXTENSIONS:

            # Call the trim_audio func to trim audio from MEDIA_PATH
            self.trim_video(
                start_time=start_time,
                end_time=end_time,
                media_presigned_uri=media_presigned_uri,
                video_path=video_path,
            )

            return audio_path, video_path
        else:
            # Assign video_path value to empty string if the extension type is of audio format
            video_path = ""
            return audio_path, video_path

    def diarize(self):
        try:
            logger.info("Diarization process has been started")
            start_index = 0

            # Initializes response variable to hold the diarization segment dict
            response = []

            # Loads the speech to text output from S3 into variable for diarization
            transcribe_response = self.s3_utils.read_json(
                read_path=self.settings.S2T_OUTPUT_PATH
            )

            # Generates list of segments from transcribe response
            segments_list = (
                transcribe_response.get("results").get("speaker_labels").get("segments")
            )

            # Iterates over list of items for getting segment wise transcript
            items_list = transcribe_response.get("results").get("items")
            for count, diarized_segment in enumerate(segments_list, start=0):
                speaker_id = diarized_segment.get("speaker_label")
                start_time = float(diarized_segment.get("items")[0].get("start_time"))
                end_time = float(diarized_segment.get("items")[-1].get("end_time"))
                start_index, transcript = get_segment_transcript(
                    start_index=start_index,
                    start_time=start_time,
                    end_time=end_time,
                    items_list=items_list[start_index:],
                )

                # Trims audio/video files based on start time and end time
                audio_path, video_path = self.trim(
                    start_time=start_time, end_time=end_time
                )

                # Creates segment dictionary for each item and appends it to response variable
                segment_dict = initilaize_segment_dict(
                    speaker_id=speaker_id,
                    speaker_label=count,
                    start_time=start_time,
                    end_time=end_time,
                    audio_path=audio_path,
                    video_path=video_path,
                    transcript=transcript,
                    extension=self.extension,
                )
                response.append(segment_dict)

            # Returns diarization response which will be stored in S3 Bucket
            return constants.DIARIZATION_SUCCESS, response

        except (RuntimeError, MemoryError, ValueError, TypeError):
            logger.error("Exception occured while running speaker diarization engine")
            return constants.DIARIZATION_FAIL, constants.DIARIZATION_ERROR_RESPONSE
