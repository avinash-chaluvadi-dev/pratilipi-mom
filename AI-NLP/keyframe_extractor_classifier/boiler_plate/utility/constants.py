APP_CONFIG_PATH = {
    "specs": "module/specs.yaml",
    "log": "logging_hook/cnfg_logging_hook.yml",
    "input": "intake_endpoint/cnfg_intake_endpoint.yml",
    "output": "output_endpoint/cnfg_output_endpoint.yml",
}

MODEL_NAMES = {
    "framify": "Framify (Keyframe Extractor)",
    "keyframecls": "keyframe_classifier",
}
FRAMIFY = "framify"
KEYFRAME_CLS = "keyframecls"

## ML serve specs
DATA_KEY = "data"
MODEL_KEY = "model"
STATUS_KEY = "status"
ERROR_KEY = "error"
SUCCESS_KEY = "success"
RESPONSE_KEY = "response"

# S3 specs/constants
S3_URI = "s3://"

# DB Status
ERROR_DB = "Error"
CANCELLED_DB = "Cancelled"

RESPONSE = {
    "framify": "Framify (Keyframe Extractor)",
    "keyframecls": "keyframe_classifier",
}

API_BASE_URL = "http://localhost:8001"

# pratilipi-tool-dashboard:1
API_DATA_CONSOLIDATE_INDEXPOINTS = {
    "ror": "Ready For Reviews",
    "mir": "MoM In reviews",
    "mgn": "MoM generated",
}

# Job States
ERROR_STATUS = "ERROR"
COMPLETED_STATUS = "COMPLETED"
IN_PROGRESS_STATUS = "IN_PROGRESS"
JOB_NOT_FOUND_STATUS = "JOB_NOT_FOUND"

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

APP_ROLE_TYPES = ["admin"]
