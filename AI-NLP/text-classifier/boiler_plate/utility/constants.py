# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Global Constant File***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

APP_CONFIG_PATH = {
    "input": "intake_endpoint/cnfg_intake_endpoint.yml",
    "output": "output_endpoint/cnfg_output_endpoint.yml",
    "specs": "module/specs.yml",
    "log": "cnfg_logging_hook.yml",
}

API_BASE_URL = "http://localhost:8001"

# Ml model names for serve response
MODEL_NAMES = {
    "ner": "Spacy_NER",
    "allocator": "allocator",
    "labelcls": "label_classifier",
    "markercls": "marker_classifier",
    "sentiment": "sentiment_classifier",
    "escalation": "escalation_classifier",
}
NER = "ner"
LABEL_CLS = "labelcls"
MARKER_CLS = "markercls"
ALLOCATOR_CLS = "allocator"
SENTIMENT_CLS = "sentiment"
ESCALATION_CLS = "escalation"

# Response Keys
RESPONSE = {
    "ner": "Spacy_NER",
    "allocator": "allocator",
    "labelcls": "label_classifier",
    "markercls": "marker_classifier",
    "sentiment": "sentiment_classifier",
    "escalation": "escalation_classifier",
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

ANNOTATION_ADAPTOR_RULES_PATH = "module/feedback_adapter/data/rules.json"
