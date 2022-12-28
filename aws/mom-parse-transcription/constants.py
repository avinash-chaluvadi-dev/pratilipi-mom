# Speech to text specs
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

# AWS Transcribe params
FAILED = "FAILED"
RESULT = "results"
COMPLETED = "COMPLETED"
TRANSCRIPT = "transcript"
TRANSCRIPTS = "transcripts"

# S3 Params
S3_URI = "s3://"

# Response params
RESPONSE = "response"

# Diarization specs
EXTENSION_KEY = "extension"
DIARIZATION_KEY = "diarization"

# Constants for Status
ERROR = "error"
STATUS = "status"
SUCCESS = "success"

# Constants for SNS Success message
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
