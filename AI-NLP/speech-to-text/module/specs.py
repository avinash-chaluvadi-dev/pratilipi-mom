from boiler_plate.utility.read_yaml import ReadYaml


class Specs:
    def __init__(self, path):
        self.path = path
        yaml = ReadYaml(self.path)
        self.specs = yaml.get_yaml()

    def get_input_location(self):
        return self.specs["speech_to_text"]["input"]

    def get_output_location(self):
        return self.specs["speech_to_text"]["output"]
