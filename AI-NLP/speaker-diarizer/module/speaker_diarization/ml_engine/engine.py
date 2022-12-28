"""

This is the engine module of the Speaker Diarizatoin Model. This module invokes the speaker recognizer and the
Diarization models for the training, serving and evaluation component.

"""
import logging
import os
import pickle
import shutil
import sys
import traceback

import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment

from .. import config
from .diarization import uisrnn
from .speaker_recognition.ghostvlad import model as spkModel
from .speaker_recognition.ghostvlad import toolkits

logging.basicConfig(
    filename=config.LOG_RESULTS_DIR + config.LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
)


class DiarizeEngine:
    """
    This Engine class is the pluggable interface for the training component of the UISRNN network and the serving of
    the speaker diarization. The serving component to be run in two modes, one is inference, which can be plugged into
    the application API. The other is the save mode, which is to be run in the standalone execution mode.

    """

    def __init__(self):
        # This piece of code will be converted once, we refactor the training component code base here.
        pass

    def train_uisrnn(self):

        # BELOW CODE BASE NEEDS SOURCE CODE RECFACTORING, WILL BE DONE IN THE NEXT ITERATION[line 40 - line 48].
        # Instantiate the UISRNN MODEL.
        """
        model = uisrnn.UISRNN(config.UISRNN["MODEL"])

        # Load the training data.
        train_data = np.load(config.UISRNN["TRAIN"]["PATH"])
        train_sequence_list = [seq.astype(float) + 0.00001 for seq in train_data["train_sequence"]]
        train_cluster_id_list = [np.array(cid).astype(str) for cid in train_data["train_cluster_id"]]
        model.fit(train_sequence_list, train_cluster_id_list, config.UISRNN["TRAIN"])
        model.save(config.UISRNN["TRAIN"]["SAVE"])
        """

    def serve(self, audio_file, mode="infer"):
        """
        parameters : audio_file -> path to the audio recording for which to perform the diarization.

        return : The response is conditional JSON file, if the mode == "infer", it return a json in the expected
                 format to be consumed by the API team, else if the mode == "save" the results are logged in the
                 results folder,since, that is done for the standalone testing component.
        """

        try:
            if mode == "infer":
                # Return the chunks of the diarized files byte file.
                # Check the code base from sentiment classifier and return a response.
                serial_audio_chunks, serial_df = diarize(
                    wav_path=audio_file,
                    embedding_per_second=config.SERVE["EMBEDDING_PER_SECOND"],
                    overlap_rate=config.SERVE["OVERLAP_RATE"],
                )

                # Format the response JSON
                response = {
                    "Speaker_Diarizer_Result": {
                        "status": "Success",
                        "Details": {
                            "audio_chunks": serial_audio_chunks,
                            "dataframe": serial_df,
                        },
                    }
                }
                return response

            elif mode == "save":
                # Keep this similar to the Diarize method itself.
                # store the files, in the results and the csv too.
                # The return here is obselete, since it is not getting consumed anywhere
                # Adding return here, to abide with the PEP-8 conventions.
                chunks_df = diarize(
                    wav_path=audio_file,
                    store_seg_dir=config.SEG_RESULTS_DIR,
                    embedding_per_second=config.SERVE["EMBEDDING_PER_SECOND"],
                    overlap_rate=config.SERVE["OVERLAP_RATE"],
                )
                return chunks_df

            else:
                raise ValueError("Invalid mode for the serve method")

        except Exception:
            # Any exception occurs, handle it by returning the response.
            # Add the traceback to the Details key when the serving process fails.
            logging.exception(
                "Exception encountered while serving the Sentiment Engine",
                exc_info=True,
            )
            response = {
                "Speaker_Diarizer_Result": {
                    "status": "Fail",
                    "Details": f"{traceback.format_exc()}",
                }
            }
            return response


# Speaker Diarization Utilities.
def diarize(
    wav_path=None,
    store_seg_dir=None,
    embedding_per_second=None,
    overlap_rate=None,
):
    """
    This is the main diarization function, which will be invoked inside the engine's serve method.
    inputs : wav_path -> The path to the input audio file/video file of the meeting.
    store_seg_dir -> The path to the storage of the output from the diarization.
    embedding_per_second ->

    return : if store_seg_dir provided, stores the generated csv file and audio chunks in the results/store_segs.
             else, serialized audio file chunks objects and serialized dataframe are returned.
    """

    # gpu configuration.
    toolkits.initialize_gpu(config.SERVE["DEVICE"])

    #  Speaker Recognition Network Instantiation
    logging.debug("Parameters are loaded ..")
    network_eval = spkModel.vggvox_resnet2d_icassp(
        input_dim=config.SERVE["SAMPLE_PARAMS"]["DIM"],
        num_class=config.SERVE["SAMPLE_PARAMS"]["N_CLASSES"],
        mode="eval",
        args=config,
    )
    logging.debug("VGGVOX_resnet2d_icassp network loading ..")
    network_eval.load_weights(config.SERVE["VGG_MODEL_PATH"], by_name=True)
    logging.debug("Weights are loaded ..")

    # UISRNN - Diarisation Network Instantiation
    model_args, inference_args = (
        config.UISRNN["MODEL"],
        config.UISRNN["INFERENCE"],
    )
    model_args["OBSERVATION_DIM"] = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    logging.debug("UISRNN Model Loading ..")
    uisrnnModel.load(config.SERVE["UISRNN_MODEL_PATH"])

    # Data Loading
    specs, intervals = load_data(
        wav_path,
        embedding_per_second=embedding_per_second,
        overlap_rate=overlap_rate,
    )
    logging.debug("Audio File Loaded ..")

    # GenMAP
    mapTable, keys = genMap(intervals)
    logging.debug("Generated Map ..")

    # Get the Speaker Recogntion Prediction -> The embedding vector for speakers.
    feats = []
    for idx, spec in enumerate(specs):
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        logging.debug(f"{idx} : Prediction vggvox model")
        v = network_eval.predict(spec)
        feats += [v]

    # Create Feature Embeddings
    feats = np.array(feats)[:, 0, :].astype(float)  # [splits, embedding dim]

    # Passing the Feature Embedding Created from the Speaker Recognition System to the Diarization predictor
    logging.debug("Prediction UISRNN Model ..")
    predicted_label = uisrnnModel.predict(feats, inference_args)

    logging.debug("Time_spec_rate ..")
    # The time_spec_rate > Speaker Embedding extracted tsr every ms
    time_spec_rate = 1000 * (1.0 / embedding_per_second) * (1.0 - overlap_rate)
    center_duration = int(1000 * (1.0 / embedding_per_second) // 2)

    # Based on the time_spec_rate, arrange results
    speakerSlice = arrangeResult(predicted_label, time_spec_rate)

    logging.debug("Time map to orign wav ..")
    for (
        spk,
        timeDicts,
    ) in speakerSlice.items():  # time map to orgin wav(contains mute)
        for tid, timeDict in enumerate(timeDicts):
            s = 0
            e = 0
            for i, key in enumerate(keys):
                if s != 0 and e != 0:
                    break
                if s == 0 and key > timeDict["start"]:
                    offset = timeDict["start"] - keys[i - 1]
                    s = mapTable[keys[i - 1]] + offset
                if e == 0 and key > timeDict["stop"]:
                    offset = timeDict["stop"] - keys[i - 1]
                    e = mapTable[keys[i - 1]] + offset

            speakerSlice[spk][tid]["start"] = s
            speakerSlice[spk][tid]["stop"] = e

    logging.debug("Audio Segmented from the wav ..")
    audio = AudioSegment.from_wav(wav_path)

    # Run in the storage mode i.e. testing mode.
    if store_seg_dir:
        # Create DataFrame to store the diarized Chunks to the CSV
        df = pd.DataFrame(
            {
                "wav_filename": [],
                "wav_filesize": [],
                "start_time": [],
                "stop_time": [],
                "label": [],
            }
        )
        cols = [
            "wav_filename",
            "wav_filesize",
            "start_time",
            "stop_time",
            "label",
        ]

        wave_file_name = wav_path.split("/")[-1].split(".wav")[0]

        # Calling the greedy decoder for the segments generation
        # Create a sub-folder under store segs corresponding to a source wave file.
        audio_seg_store = store_seg_dir + "/" + wave_file_name

        # Create the sub-folder, if does not exist.
        if not os.path.exists(audio_seg_store):
            os.mkdir(audio_seg_store)

        # If already exists, rewrite the folder.
        else:
            shutil.rmtree(audio_seg_store)
            os.mkdir(audio_seg_store)

        for spk, timeDicts in speakerSlice.items():
            logging.debug("========= " + str(spk) + " =========")
            # Below we get the diarization dataframe and the segements audio file object dictionary.
            _, df = greedy_combine_segs(
                "save", audio, timeDicts, spk, df, cols, audio_seg_store
            )

        df = df.sort_values(
            ["start_time", "stop_time"], ascending=[True, True]
        )

        # Storing the generated diarized csv to the wave_file_name sub-folder
        diarize_csv = store_seg_dir + "/" + wave_file_name + "/diarize.csv"
        logging.debug("Saving the Chunks to the CSV ..")
        df.to_csv(diarize_csv, index=False)
        logging.debug("Speaker Diarizaion Completed ..")

        return diarize_csv

    else:
        # Create DataFrame to store the diarized Chunks to the CSV
        df = pd.DataFrame(
            {
                "wav_filename": [],
                "start_time": [],
                "stop_time": [],
                "label": [],
            }
        )
        cols = ["wav_filename", "start_time", "stop_time", "label"]
        logging.debug("Speaker_Slice ..")

        # Dictionary to be used to
        segment_dict = {}
        for spk, timeDicts in speakerSlice.items():
            logging.debug("========= " + str(spk) + " =========")
            # Below we ge the diarization dataframe and the segements audio file object dictionary.
            spk_audio, df = greedy_combine_segs(
                "infer", audio, timeDicts, spk, df, cols
            )
            segment_dict[spk] = spk_audio

        df = df.sort_values(
            ["start_time", "stop_time"], ascending=[True, True]
        )
        logging.debug("Speaker Diarizaion Completed ..")

        return segment_dict, pickle.dumps(df)


def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    time_dict = {"start": int(value[0] + 0.5), "stop": int(value[1] + 0.5)}
    if key in speakerSlice:
        speakerSlice[key].append(time_dict)
    else:
        speakerSlice[key] = [time_dict]

    return speakerSlice


def arrangeResult(labels, time_spec_rate):
    # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i, label in enumerate(labels):
        if label == lastLabel:
            continue
        speakerSlice = append2dict(
            speakerSlice, {lastLabel: (time_spec_rate * j, time_spec_rate * i)}
        )
        j = i
        lastLabel = label
    speakerSlice = append2dict(
        speakerSlice,
        {lastLabel: (time_spec_rate * j, time_spec_rate * (len(labels)))},
    )
    return speakerSlice


def genMap(intervals):  # interval slices to maptable
    slicelen = [sliced[1] - sliced[0] for sliced in intervals.tolist()]
    mapTable = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        mapTable[idx] = sliced[0]
        idx += slicelen[i]
    mapTable[sum(slicelen)] = intervals[-1, -1]

    keys = [k for k, _ in mapTable.items()]
    keys.sort()
    return mapTable, keys


def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond % 1000
    minute = timeInMillisecond // 1000 // 60
    second = (timeInMillisecond - minute * 60 * 1000) // 1000
    time = "{}:{:02d}.{}".format(minute, second, millisecond)
    return time


def load_wav(vid_path, sr):
    """
    input : vide_path -> the path to the input audio file.
    sr : the sample rate at which to load the wave file.

    return : numpy array(list of wav_output), list of intervals for all the slices of the audio file that have been
    created after the split.
    """
    wav, _ = librosa.load(vid_path, sr=sr)
    # The top_db decides, what decible of voide should be considered as the silence threshold.
    # Default is 60, we are using 20, being careful about catching the low noises.
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0] : sliced[1]])
    return np.array(wav_output), (intervals / sr * 1000).astype(int)


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    """
    refer the link : librosa.org/doc/main/generated/librosa.stft.html, for the details of
    Short Time Fourier Transform[STFT]
    """
    linear = librosa.stft(
        wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )  # linear spectrogram
    return linear.T


def load_data(
    path,
    win_length=400,
    sr=20000,
    hop_length=160,
    n_fft=512,
    embedding_per_second=0.5,
    overlap_rate=0.5,
):
    """
    parameters: path -> the path to the audio meeting file.
                win_length -> to be used to create linear spectogram for a stride of audio file.
                sr -> sampling rate at which the audio should be loading.[human perceivable sampling
                rate is 20Khz-40Khz]
                hop_length ->  step size, the gap between the two frames of audio for which you create
                linear_spectrogram.
                n_fft-> total audio sample to be used for one spectrogram window.


    """
    wav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    # Spectrogram length - sampling rate/ hop_length = total number of steps to complete 1 sec of signal.
    # total number of steps to complete 1 sec of signal * embedding per sec
    spec_len = sr / hop_length / embedding_per_second
    spec_hop_len = spec_len * (1 - overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while True:  # slide window.
        if cur_slide + spec_len > time:
            break
        spec_mag = mag_T[
            :, int(cur_slide + 0.5) : int(cur_slide + spec_len + 0.5)
        ]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec, intervals


def store_segments(
    mode=None, audio=None, ts=None, te=None, label=None, store_dir=None
):

    audio_chunk = audio[ts:te]

    if mode == "save":
        filepath = store_dir + "/" + "chunk_{}_{}_{}.wav".format(ts, te, label)
        # Saving the audio chunk.
        audio_chunk.export(filepath, format="wav")
        # Don't know what is ffmpeg doing here.
        os.system(
            "ffmpeg -loglevel panic -y -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}".format(
                filepath, filepath
            )
        )
        return filepath, os.path.getsize(filepath)

    else:
        filepath = "chunk_{}_{}_{}.wav".format(ts, te, label)
        # Serializing the audio file as pickle object - for API response artifact.
        audio_dict = {
            filepath: pickle.dumps(
                audio_chunk, protocol=pickle.HIGHEST_PROTOCOL
            )
        }
        return audio_dict, filepath


def greedy_combine_segs(
    mode=None,
    audio=None,
    time_dicts=None,
    spk=None,
    df=None,
    cols=None,
    store_dir=None,
):

    i = 0
    rows = []
    chunk_dict = {}
    while i < len(time_dicts):
        j = i + 1
        while (
            j < len(time_dicts)
            and (time_dicts[j]["start"] / 1000)
            - (time_dicts[j - 1]["stop"] / 1000)
            <= config.SERVE["DIFF_DURATION"]
        ):
            j += 1
        j -= 1

        start_time = time_dicts[i]["start"]
        end_time = time_dicts[j]["stop"]

        # Remove Unnecessary Voice.
        if (
            end_time / 1000 - start_time / 1000
            < config.SERVE["CUT_TH_DURATION"]
        ):
            i = j + 1
            continue

        # Conditional execution of the store segments.
        if mode == "save":
            fn, fs = store_segments(
                mode="save",
                audio=audio,
                ts=start_time,
                te=end_time,
                label=spk,
                store_dir=store_dir,
            )
            rows.append([fn, fs, start_time, end_time, spk])
        else:
            seg_dict, fn = store_segments(
                mode="infer",
                audio=audio,
                ts=start_time,
                te=end_time,
                label=spk,
            )
            chunk_dict.update(seg_dict)
            rows.append([fn, start_time, end_time, spk])
        s = fmtTime(start_time)
        e = fmtTime(end_time)
        logging.debug(s + " ==> " + e)
        i = j + 1
    df_temp = pd.DataFrame(rows, columns=cols)

    return chunk_dict, df.append(df_temp, ignore_index=True)
