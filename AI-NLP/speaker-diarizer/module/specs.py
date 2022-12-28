from boiler_plate.utility.read_yaml import ReadYaml


class Specs:
    """Python object repr of specs.yml file"""

    def __init__(self, path: str):
        """
        path: path of the yaml file which needs to be parsed
        """
        self.path = path
        yaml = ReadYaml(self.path)
        self.specs = yaml.get_yaml()

    def get_input_location(self) -> str:
        return self.specs["speaker_diarization"]["input"]

    def get_output_location(self) -> str:
        return self.specs["speaker_diarization"]["output"]
