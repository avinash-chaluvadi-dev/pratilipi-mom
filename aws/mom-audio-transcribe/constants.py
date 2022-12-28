# Speech to Text specs
SPEECH_TO_TEXT = {
    "model_key": "model",
    "model_val": "speech_to_text",
    "output_dir": "speech_to_text/output",
    "output_file": "speech_to_text.json",
}
ERROR_SUBJECT = "Speech To Text Error Response"
S2T_SUBJECT = "Speech To Text Success Response"

# Diarization specs
DIARIZATION = {
    "model_key": "model",
    "model_val": "speaker_diarization",
    "output_file": "speaker_diarization.json",
    "output_dir": "speaker_diarization/output",
}
# Diarization specs
EXTENSION_KEY = "extension"
DIARIZATION_KEY = "diarization"

# Transcribe specs/constants
MESSAGE_KEY = "message"
TRANSCRIBE_JOB = "transcribe_job_name"
TRANSCRIBE_FAIL = False
TRANSCRIBE_SUCCESS = True


# Speech To Text Static paths
# BASE_PATH = "output"
# STORE_PATH = "speech_to_text/output/speech_to_text.json"
S2T_DEFAULT_FILE = "speech_to_text.json"
TRANSCRIBE_JOB_NAME = "pratilipi_transcription"

# S3 Specs
S3_URI = "s3://"

# Constants for status
ERROR = "error"
STATUS = "status"
SUCCESS = "success"
RESPONSE = "response"

# Constants for SNS message
MEDIA = "media_path"
S2T_OUTPUT = "s2t_output_path"
DIARIZATION_OUTPUT = "diarization_output_path"

# Summarizer MS specs
STATUS_UPDATE_SUFFIX = "api/file/status-update/"
STATUS = "status"
FILE_NAME = "file"

# DB related constants
ERROR_STATUS = "Error"
PROCESSING_STATUS = "Processing"
