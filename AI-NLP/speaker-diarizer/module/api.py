"""
Dummpy API script to test the serving component of the package integrates with the API.

"""
import pickle
from pathlib import Path

# from moviepy.video.io.VideoFileClip import VideoFileClip
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pydub import AudioSegment
from speaker_diarization import spk_diarize_serve


def covert_mp4_to_slice(resp):
    segment_dict = pickle.loads(resp["Speaker_Diarizer_Result"]["Details"]["dataframe"])

    for i in range(len(segment_dict["start_time"])):
        ffmpeg_extract_subclip(
            "C://General//Hive Workspace/SMS_speaker_diarization/Sprint13Rec/HIVE_20210824_121621.mp4",
            segment_dict["start_time"][i],
            segment_dict["stop_time"][i],
            targetname=f"C://Users//AG99023//Downloads//sample_{segment_dict['start_time'][i]}_{segment_dict['stop_time'][i]}.mp4",
        )


def convert_mp3_to_wav():
    src = "C://General//Hive Workspace/SMS_speaker_diarization/Sprint13Rec/HIVE_20210824_121621.mp4"

    wav_src = "C://General//Hive Workspace//SMS_speaker_diarization//file.wav"
    # print(src)
    sound = AudioSegment.from_file(src)
    # print(sound)
    sound.set_frame_rate(16000)
    input_file = sound.export(wav_src, format="wav")
    # print(input_file)
    resp = spk_diarize_serve(input_file)
    covert_mp4_to_slice(resp)

    segment_dict = resp["Speaker_Diarizer_Result"]["Details"]["audio_chunks"]
    Path(f"C://General//Hive Workspace//SMS_speaker_diarization//211021/").mkdir(parents=True, exist_ok=True)
    for spk in segment_dict.keys():
        # print("spk : ", spk)
        for chunk_name in segment_dict[spk].keys():
            audio_file = pickle.loads(segment_dict[spk][chunk_name])
            audio_file.export(Path(f"C://General//Hive Workspace//SMS_speaker_diarization//211021/{chunk_name}"),format="wav")


if __name__ == "__main__":
    convert_mp3_to_wav()

    # print(resp["Speaker_Diarizer_Result"]["status"])
    # response_decoder(resp)
