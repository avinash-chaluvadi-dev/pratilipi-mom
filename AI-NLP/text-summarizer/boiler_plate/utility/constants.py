APP_CONFIG_PATH = {
    "input": "intake_endpoint/cnfg_intake_endpoint.yml",
    "output": "output_endpoint/cnfg_output_endpoint.yml",
    "specs": "module/specs.yml",
    "log": "cnfg_logging_hook.yml",
    "jira": "boiler_plate/corejira/Jira_configuration.yml",
}

# S3 constants/specs
S3_URI = "s3://"
MOM_UPLOAD_DIR = "mom"
API_BASE_URL = "http://localhost:8001"
LOCAL_HOST_URL = "http://localhost:8000/media/"

ANNOTATION_ADAPTOR_RULES_PATH = "module/feedback_adapter/data/rules.json"

# Transcribe - Speaker Diarization and Speech To Text
SPEECH_TO_TEXT = "speech_to_text/output/speech_to_text.json"
SPEAKER_DIARIZATION = "speaker_diarization/output/speaker_diarization.json"

# Classifier Output Paths
NER = "ner/output/ner_output.json"
DEADLINE_ESCALATION = "allocator/output/allocator_output.json"
LABEL_CLASSIFIER = "label_classifier/output/label_output.json"
MARKER_CLASSIFIER = "marker_classifier/output/marker_output.json"
SENTIMENT_CLASSIFIER = "sentiment_classifer/output/sentiment_output.json"
RECOMMENDATION_CLASSIFIER = "recommendation/output/recommendation_output.json"

# Summarizer Output Paths
SUMMARIZER = "summarizer/output/summarizer_output.json"
HEADLINER = "headliner/output/headliner_output.json"

# Keyframes Output Paths
KEYFRAME_EXTRACTOR = "keyframe_extractor/output/keyframe_ext_output.json"
KEYFRAME_CLASSIFIER = "keyframe_classifier/output/keyframe_cls_output.json"

MODELS = [
    "NER",
    "LABEL_CLASSIFIER",
    "SENTIMENT_CLASSIFIER",
    "SUMMARIZER",
    "HEADLINER",
    "KEYFRAME_EXTRACTOR",
    "KEYFRAME_CLASSIFIER",
]

# Model dictionary for status_check endpoint
HEADLINER_MODEL = "headliner"
SUMMARIZER_MODEL = "summarizer"
RESPONSE = {"summarizer": "summarizer", "headliner": "headliner", "mom": "mom"}
MODEL_NAMES = {
    "summarizer": "summarizer",
    "headliner": "headliner",
    "mom": "mom",
    "framify": "Framify (Keyframe Extractor)",
    "keyframecls": "keyframe_classifier",
}

## ML serve specs
DATA_KEY = "data"
MODEL_KEY = "model"
STATUS_KEY = "status"
ERROR_KEY = "error"
SUCCESS_KEY = "success"
RESPONSE_KEY = "response"

# Job States
ERROR_STATUS = "ERROR"
FAILED_STATUS = "FAILED"
COMPLETED_STATUS = "COMPLETED"
IN_PROGRESS_STATUS = "IN_PROGRESS"
JOB_NOT_FOUND_STATUS = "JOB_NOT_FOUND"

# DB Status
ERROR_DB = "Error"
CANCELLED_DB = "Cancelled"

# apscheduler - Jobstore params
JOB_STORE_URL = "url"
JOB_STORE_KEY = "type"
JOB_STORE_TYPE = "sqlalchemy"
JOB_STORE = "apscheduler.jobstores.default"

JOB_STORES = {JOB_STORE_TYPE: {JOB_STORE_URL: "sqlite:///jobs.sqlite"}}

# apscheduler - Executor params
EXECUTOR_KEY = "class"
EXECUTOR_VALUE = "value"
EXECUTOR_TYPE = "ThreadPoolExecutor"
EXECUTOR = "apscheduler.executors.default"

# apscheduler - Worker params
MAX_WORKERS = "max_workers"

EXECUTORS = {
    EXECUTOR_TYPE: {
        MAX_WORKERS: 12,
        EXECUTOR_VALUE: "apscheduler.executors.pool:ThreadPoolExecutor",
    }
}

# apscheduler - coalesce params
COALESCE_VALUE = "false"
COALESCE_KEY = "apscheduler.job_defaults.coalesce"

# apscheduler - Event codes
EVENT_FAIL = 8192
EVENT_SUCCESS = 4096

# Role params
SME = "sme"
ADMIN = "admin"

# Dashboard params
DASHBOARD_DATE_FORMAT = "YYYY-MM-DD"

# Data Filter specs
FILTER_LABELS = "Labels"
FILTER_ENTITIES = "Entities"
FILTER_SENTIMENTS = "Sentiments"
FILTER_PARTICIPANTS = "Participants"
FILTER_FLATENNED_LIST = "flattened_list"
ENTITY_UI_MODEL_MAP = {
    "Date": "Date",
    "Tool": "Tool",
    "Event": "Event",
    "Scrum": "Scrum",
    "Status": "Status",
    "Name": "Person Name",
    "Team name": "Team Name",
    "Technology": "Technologies",
    "Anthem tool": "Anthem Tools",
    "Organization": "Organization",
    "Technical platform": "Technical Platform",
}

# MOM specs
MOM_STRING = [
    "date",
    "summary",
    "end_time",
    "assign_to",
    "audio_path",
    "manual_add",
    "start_time",
    "video_path",
    "transcript",
    "speaker_id",
    "speaker_label",
]
MOM_LIST = ["keyframe_extractor", "keyframe_labels"]
MOM_DICT = ["label", "marker", "bkp_label", "sentiment", "bkp_sentiment"]

MOM_LABEL = "label"
MOM_BKP_LABEL = "label"
MOM_SUMMARY = "summary"
MOM_ENTITY_TYPE = "type"
MOM_ENTITIES = "entities"
MOM_SENTIMENT = "sentiment"
MOM_ENTRIES = "mom_entries"
MOM_MANUAL_CONFIDENCE = 100
MOM_TRANSCRIPT = "transcript"
MOM_MANUAL_ADD = "manual_add"
MOM_SPEAKER_ID = "speaker_id"
MOM_MANUAL_DELETE = "manual_remove"
MOM_SPEAKER_LABEL = "speaker_label"
MOM_MANUAL_ENTRIES = "manual_entries"
MOM_CONCATENATED_VIEW = "concatenated_view"
MOM_ENTITIES_DEFAULT = [{"words": [], "type": []}, []]

# File upload specs
SUPPORTED_FILE_TYPES = ["mp4", "mp3", "wav"]
SUPPORTED_MIME_TYPES = ["video/mp4", "audio/mpeg", "audio/x-wav"]

# Keyframe specs
FRAMIFY = "framify"
KEYFRAMES = "keyframes"
KEYFRAME_CLS = "keyframecls"

API_DATA_CONSOLIDATE_INDEXPOINTS = {
    "ror": "Ready For Reviews",
    "mir": "MoM In reviews",
    "mgn": "MoM generated",
}
