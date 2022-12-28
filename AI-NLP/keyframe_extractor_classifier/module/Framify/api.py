import json
import os
import time

import config

from __init__ import model_serve


def load_json():
    path = (
        "/var/s3fs-demofs/Pratilipi/AI/keyframe-extractor/keyframe_extractor_input.json"
    )
    with open(path) as file:
        json_data = json.load(file)
    return json_data


def api_integration(test_input):
    return model_serve(test_input, None, config.HOSTNAME_KEYFRAMES, None)


if __name__ == "__main__":
    st = time.time()
    TEST_INPUT = load_json()
    output = api_integration(TEST_INPUT)
    print(output)
    with open(
        os.path.join(config.OUTPUT_RESULTS, "keyframe_extraction_output.json"), "w"
    ) as f:
        json.dump(output, f)
    print(f"Time taken {time.time() - st}")
