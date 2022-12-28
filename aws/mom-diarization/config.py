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
    BIT_RATE: str
    AUDIO_PATH: str
    VIDEO_PATH: str
    MEDIA_PATH: str
    BUCKET_NAME: str
    S2T_OUTPUT_PATH: str
    SIGNED_URL_TIMEOUT: int

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{
                k: v
                for k, v in env.items()
                if k in set([f.name for f in dataclasses.fields(cls)])
            }
        )
