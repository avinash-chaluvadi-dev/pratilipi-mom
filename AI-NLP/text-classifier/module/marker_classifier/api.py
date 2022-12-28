import json

from .__init__ import model_serve


def api_integration(test_input):
    return model_serve(json_data=test_input)


if __name__ == "__main__":
    path = "C:/Users/AG84959/Downloads/Pratilipi/marker_classifier/dataset/batch_serve.json"
    with open(path, "r") as f:
        json_data = json.load(f)
        output = api_integration(test_input=json_data)
