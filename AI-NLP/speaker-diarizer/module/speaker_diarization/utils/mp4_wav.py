import os
import re
from datetime import datetime

from pydub import AudioSegment


def get_duration(secs):
    mins = secs // 60
    secs = secs % 60
    hrs = mins // 60
    mins = mins % 60
    return str(int(hrs)) + " hr, " + str(int(mins)) + " min, " + str(int(secs)) + " sec"


def convert_mp42wav(filepath, meeting_type):
    BASE_STORE_PATH = "../dataset/"
    filename = filepath.split("/")[-1]
    pattern = re.compile(r"\d\d\d\d\d\d\d\d")
    match = re.findall(pattern, filename)
    if len(match) > 0:
        new_filepath = os.path.join(
            BASE_STORE_PATH, "hive_{}_".format(meeting_type) + match[0] + ".wav"
        )
        date = match[0]
    else:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        new_filepath = os.path.join(
            BASE_STORE_PATH, "hive_{}_".format(meeting_type) + date + ".wav"
        )
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_frame_rate(16000)
    audio_duration = get_duration(audio.duration_seconds)
    audio.export(new_filepath, format="wav")
    return new_filepath, date


if __name__ == "__main__":
    convert_mp42wav("C:\\Users\\AG84959\\OneDrive - Anthem\\Recordings\\KT Sessions with Amit - ML Models Packaging Convention-20211116_140258-Meeting Recording.mp4", "technical")
