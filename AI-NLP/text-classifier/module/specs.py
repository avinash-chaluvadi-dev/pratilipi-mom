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

    def get_ner_specs(self) -> str:
        return self.specs["ner"]

    def get_allocator_specs(self) -> str:
        return self.specs["allocator"]

    def get_recommendation_specs(self) -> str:
        return self.specs["recommendation"]

    def get_marker_specs(self) -> str:
        return self.specs["marker"]

    def get_label_specs(self) -> str:
        return self.specs["label"]

    def get_sentiment_specs(self) -> str:
        return self.specs["sentiment"]

    def get_escalation_specs(self) -> str:
        return self.specs["escalation"]["input"]
