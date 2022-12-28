from __init__ import model_serve

from .utils import utils_tools

# Sample test JSON to test the package.
TEST_INPUT = {
    "model_name": "speech_to_text",
    "response": [
        {
            "chunk_id": 1,
            "start_time": 333,
            "end_time": 8254,
            "speaker_id": 0,
            "path": "dataset/hive_standup_20210804_audio_chunk_4828_19024_0.wav",
            "transcript": "So let's just start with Abinash. So, anyways, as of now we are following this uh particular table, but once the scrum and everything is done, then we will be following the exact scrum rules.",
        },
        {
            "chunk_id": 2,
            "start_time": 8254,
            "end_time": 12046,
            "speaker_id": 1,
            "path": "dataset/hive_standup_20210804_audio_chunk_21592_24736_1.wav",
            "transcript": "So if this term is present, then it is definitely talking about this particular topic. So like that you guys have to start then coming to Bhanu and anuj you guys have to extract those themes. So questions. So I guess I have added in my manual thing, I have added.",
        },
        {
            "chunk_id": 3,
            "start_time": 12046,
            "end_time": 19439,
            "speaker_id": 0,
            "path": "dataset/hive_standup_20210804_audio_chunk_24736_44808_2.wav",
            "transcript": "have you uploaded it in the teams Yes. it's okay I have I have actually I'ts there I'll check it again. I I'll check it again if it's not relevant.I'll upload okay i i'll just check it again sure no problem yeah.",
        },
    ],
}


def api_integration(test_input):
    return model_serve(test_input)


if __name__ == "__main__":
    output = api_integration(TEST_INPUT)
    # print(output)
    utils_tools.save_json(output)
