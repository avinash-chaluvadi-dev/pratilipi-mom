""" [Config Module]

This module creates a dataclass for holding the configuration
variables
"""

import dataclasses


class EnforcedDataclassMixin:
    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.type == list:
                value = value.split(",")
            setattr(self, field.name, field.type(value))
            if field.type == bool:
                value = int(value)
            setattr(self, field.name, field.type(value))


@dataclasses.dataclass
class Settings(EnforcedDataclassMixin):
    FILE_URI: str
    EXTENSION: str
    SSL_VERIFY: str
    SPEAKER_DIARIZATION: int
    DOWNSTREAM_TOPIC_ARN: str

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{
                k: v
                for k, v in env.items()
                if k in set([f.name for f in dataclasses.fields(cls)])
            }
        )
