import json
from module.speech2text import config
from module.speech2text import model_serve
from module.speech2text.utils import utils_tools


def api_integration(test_input):
    return model_serve(test_input)


if __name__ == "__main__":
    path = "/var/s3fs-demofs/Pratilipi/AI/speech-to-text/dataset/diarization_output.json"
    with open(path, "r") as f:
        json_data = json.load(f)
        output = api_integration(test_input=json_data)
        print(output)
        output = utils_tools.save_result(json_data=output, output_run=config.OUTPUT_RUN, out_json=config.OUT_JSON)
