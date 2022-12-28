from . import specs
from boiler_plate.utility.constants import APP_CONFIG_PATH

specs = specs.Specs(APP_CONFIG_PATH["specs"].format("speech2text"))
