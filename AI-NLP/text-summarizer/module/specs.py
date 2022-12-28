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

    def get_topic_extractor_specs(self) -> str:
        return self.specs["topic_extractor"]

    def get_mom_specs(self) -> str:
        return self.specs["minutes_of_meeting"]

    def get_integration_specs(self) -> str:
        return self.specs["integration"]

    def get_summarizer_specs(self) -> str:
        return self.specs["text_summarizer"]

    def get_headliner_specs(self) -> str:
        return self.specs["headliner"]

    def get_framify_specs(self):
        return self.specs["framify"]

    def get_keyframe_cls_specs(self):
        return self.specs["keyframe_classifier"]
