from boiler_plate.utility.constants import APP_CONFIG_PATH

from . import specs

specs = specs.Specs(APP_CONFIG_PATH["specs"].format("speaker_diarization"))
