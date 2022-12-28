# Speech to text specs
SPEECH_TO_TEXT = {
    "model_key": "model",
    "model_val": "speech_to_text",
    "output_dir": "speech_to_text/output",
    "output_file": "speech_to_text.json",
}

# Diarization Specs
DIARIZATION = {
    "model_key": "model",
    "model_val": "speaker_diarization",
    "output_dir": "speaker_diarization/output",
    "segment_dir": "speaker_diarization",
    "audio_segments": "audio_segments",
    "video_segments": "video_segments",
    "output_file": "speaker_diarization.json",
}
DIARIZATION_FAIL = False
DIARIZATION_SUCCESS = True
EXTENSION_KEY = "extension"
DIARIZATION_KEY = "diarization"
DIARIZATION_ERROR_RESPONSE = []
ERROR_SUBJECT = "Diarization Error Response"
DIARIZATION_SUBJECT = "Diarization Success Response"

# AUDIO_PATH = "speaker_diarization/audio_segments"
# VIDEO_PATH = "speaker_diarization/video_segments"
# STORE_PATH = "speaker_diarization/speaker_diarization.json"
DIARIZATION_DEFAULT_FILE = "speaker_diarization.json"
DIARIZATION_DEFAULT_AUDIO = "audio_segments"
DIARIZATION_DEFAULT_VIDEO = "video_segments"

# ffmpeg specs
TMP_DIR = "/tmp"
AUDIO_COMMAND_SUFFIX = "-f wav -q:a 0 -map 0:a -"
VIDEO_COMMAND_SUFFIX = "-c copy -avoid_negative_ts make_zero"
# Speech to text specs
SPEECH_TO_TEXT = {"model_key": "model", "model_val": "speech_to_text"}

# Response params
RESPONSE = "response"

# Audio/Video Segments params
S3_URI = "s3://"
AUDIO_EXTENSION = "wav"
VIDEO_EXTENSION = "mp4"
AUDIO_CHUNK = "audio_chunk"
VIDEO_CHUNK = "video_chunk"
AUDIO_CONTENT_TYPE = "audio/wav"
VIDEO_CONTENT_TYPE = "video/mp4"

# AWS Transcribe params
VIDEO_EXTENSIONS = ["mp4", "webm"]
RESULT = "results"
TRANSCRIPT = "transcript"
TRANSCRIPTS = "transcripts"

# Constants for Status
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

# Keyframe constants
KEYFRAME_EXTRACTOR_DEFAULT = []
KEYFRAME_CLASSIFIER_DEFAULT = []
KEYFRAME_CLASSIFIER = "keyframes"
KEYFRAME_EXTRACTOR = "keyframe_labels"
