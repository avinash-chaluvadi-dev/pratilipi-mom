import os

from .base import *

SECRET_KEY = env("SECRET_KEY")

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": os.getenv("DB_NAME", env("DB_NAME")),
        "USER": os.getenv("DB_USER", env("DB_USER")),
        "PASSWORD": os.getenv("DB_PASSWORD", env("DB_PASSWORD")),
        "HOST": os.getenv("DB_HOST", env("DB_HOST")),
        "PORT": int(os.getenv("DB_PORT", env("DB_PORT"))),
    }
}

ALLOWED_CIDR_NETS = [
    "10.0.0.0/8",
    "100.0.0.0/8",
]


# TODO: Update this when we are cloud
HOST = "http://localhost:3000"

CORS_ALLOWED_ORIGINS = [
    HOST,
]

SUMMARIZER_API_BASE_URL = "http://localhost:8000/"
DIARIZATION_API_BASE_URL = "http://localhost:8001/"
S2T_API_BASE_URL = "http://localhost:8002/"
CLASSIFIERS_API_BASE_URL = "http://localhost:8003/"
KEYFRAME_API_BASE_URL = "http://localhost:8004/"
