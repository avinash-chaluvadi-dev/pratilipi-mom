import json
import logging
import math
import os
from pathlib import Path
from typing import List, Union

import boto3
import cv2
import numpy as np
from scipy.signal import argrelextrema

from .. import config
from ..utils import utils_tools

if not config.USE_EFS:
    logging.basicConfig(
        filename=os.path.join(config.OUTPUT_LOG, config.LOG_FILE),
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
    )


class Frame:
    """
    Frame class to hold information about each frame
    Attributes:
        id: frame id
        frame_arr: numpy array representation of the frame
        value: difference from the previous frame

    """

    def __init__(self, id: int, frame_arr: np.ndarray, value: int) -> None:
        self.id = id
        self.frame_arr = frame_arr
        self.value = value

    def __eq__(self, o):
        return self.id == o.id

    def __ne__(self, o):
        return not self.__eq__(o)

    def __lt__(self, o):
        return self.id < o.id

    def __gt__(self, o):
        return self.id > o.id


class FramifyBackbone:
    """
    Framify Backbone class - used as a backbone model for the Framify (Keyframe extractor) package

    """

    def __init__(self) -> None:
        # Creating the keyframes directory if it's not present
        if not config.USE_S3 and not os.path.exists(config.KEYFRAMES_DIR):
            os.mkdir(config.KEYFRAMES_DIR)
            self.empty_dir()  # Clearing output directory
        self.chunk_id = "Evaluation"

    def get_histogram(self, image: np.ndarray):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 255])
        return histogram

    def empty_dir(self):
        """
        Wrapper function to clear the directory

        """
        utils_tools.empty_dir(config.KEYFRAMES_DIR)

    def get_frame_count(self, video_cap: cv2.VideoCapture) -> int:
        """
        Function to count the number of frames in a video

        Parameters:
            video_cap: Video Capture object

        Returns:
            Number of frames
        """
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return frame_count

    def _smoothen_array(
        self, array: List, window_len: int, window_type: str
    ) -> Union[list, np.ndarray]:
        """
        Smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.

        Parameters:
            array: the input signal
            window_len: the dimension of the smoothing window
            window_type: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        Returns:
            the smoothed signal

        """
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        if array.ndim != 1:
            raise ValueError("Function Smooth only accepts 1 dimensional array.")
        if array.size < window_len:
            raise ValueError("Input array needs to be bigger than window size.")
        if window_len < 3:
            return array
        if window_type not in config.SMOOTHENING_WINDOW_LIST:
            raise ValueError(
                f"Window type should be one of {config.SMOOTHENING_WINDOW_LIST}"
            )
        array = np.r_[
            2 * array[0] - array[window_len:1:-1],
            array,
            2 * array[-1] - array[-1:-window_len:-1],
        ]

        if window_type == "flat":  # moving average
            w = np.ones(window_len, "d")
        else:
            w = getattr(np, window_type)(window_len)
        final = np.convolve(w / w.sum(), array, mode="same")
        return final[window_len - 1 : -window_len + 1]

    def save_local_maxima(
        self, frames: List, frame_diffs: List, window_len: int
    ) -> List[Union[str, Path]]:
        """
        Function to save local maximum frames from a list of frames (using the frame differences)
        Parameters:
            frames: list of Frame class objects.
            frame_diffs: list of frame differences.
            window_len: length of window to be used in smoothening the frame_diffs list.

        Returns:
            List of paths of frames which were saved.

        """
        logging.info(f"Saving Frames for transcript {self.chunk_id}")
        frame_format = config.FRAME_FORMAT  # frame format (jpg or png)
        smoothen_diff_arr = self._smoothen_array(
            frame_diffs,
            window_len=window_len,
            window_type=config.DEFAULT_WINDOW_TYPE,
        )

        # calculating the relative extrema of data
        frame_indexes = np.array(argrelextrema(smoothen_diff_arr, np.greater))[0]

        file_paths = []
        for ind in frame_indexes:
            file_name = f"frame_{str(frames[ind - 1].id)}.{frame_format}"
            dir_path = os.sep.join([config.KEYFRAMES_DIR, f"Chunk {self.chunk_id}"])
            if not config.USE_S3 and not os.path.exists(dir_path):
                os.mkdir(dir_path)
            file_path = f"s3://{config.AWS_STORAGE_BUCKET_NAME}/{os.sep.join([dir_path, file_name])}"
            file_paths.append(file_path)
            # Saving the frame
            if not config.USE_S3:
                cv2.imwrite(file_path, frames[ind - 1].frame_arr)
            else:
                image_string = cv2.imencode(f".{config.FRAME_FORMAT}", img)[
                    1
                ].tostring()
                s3_utils.put_data(
                    store_prefix=s3_utils.parse_s3_uri(file_uri=file_path),
                    content_type=config.IMAGE_CONTENT_TYPE,
                    content=image_string,
                )
        return file_paths

    def save_frame(self, frame_id: int, img: np.ndarray):
        s3_utils = utils_tools.S3Utils(bucket_name=config.AWS_STORAGE_BUCKET_NAME)
        dir_path = f"{config.KEYFRAMES_DIR}Chunk {self.chunk_id}"
        file_name = f"frame_{str(frame_id)}.{config.FRAME_FORMAT}"
        file_path = f"s3://{config.AWS_STORAGE_BUCKET_NAME}/{os.sep.join([dir_path, file_name])}"
        if config.USE_S3:
            image_string = cv2.imencode(f".{config.FRAME_FORMAT}", img)[1].tostring()
            s3_utils.put_data(
                store_prefix=s3_utils.parse_s3_uri(file_uri=file_path),
                content_type=config.IMAGE_CONTENT_TYPE,
                content=image_string,
            )
            return file_path
        else:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            file_path = os.path.join(dir_path, file_name)
            cv2.imwrite(file_path, img)
            return file_path

    def run(
        self,
        input_file: Union[Path, str],
        chunk_id: int = None,
        num_transcripts: int = 1,
    ) -> List[Union[str, Path]]:
        """
        Function to extract Key frames from a given video file

        Parameters:
            input_file: file path for the input video snippet
            chunk_id: index (id) of the video snippet passed
            num_transcripts: number of videos (transcripts) in the json_data (default 1 for evaluation).
                            This is being used to limit the amount of information shown in the log file.

        Returns:
            List of frame paths saved (key frames).

        """

        # If no chunk id is provided, the video is evaluation video
        s3_utils = utils_tools.S3Utils(bucket_name=config.AWS_STORAGE_BUCKET_NAME)
        input_prefix = s3_utils.parse_s3_uri(file_uri=input_file)

        # Checking if the input file exists or not
        if not config.USE_S3 and not os.path.exists(input_file):
            logging.exception(f"{input_file} - file not present.")
            raise FileNotFoundError(f"{input_file} - file not present.")
        if config.USE_S3 and not s3_utils.prefix_exist(file_prefix=input_prefix):
            logging.exception(f"{input_file} - file not present.")
            raise FileNotFoundError(f"{input_file} - file not present.")

        if chunk_id is None:
            chunk_id = "Evaluation"
            logging.info(f"Working on Evaluation Video...")
        else:
            logging.info(f"Working on Transcript {chunk_id}...")
        self.chunk_id = str(chunk_id)

        # defining a video capture object
        s3_source_signed_uri = s3_utils.generate_presigned_uri(
            object_prefix=input_prefix
        )
        video_capture = cv2.VideoCapture(s3_source_signed_uri)

        # num of information to be shown in the current transcripts
        num_info_per_chunk = (num_transcripts * config.NUM_INFO) // num_transcripts

        # Getting number of frames for current video snippet
        num_frame = self.get_frame_count(video_capture)

        logging.info(f"NUMBER_FRAMES ----> {num_frame}")

        # Total number of frames which will be processed for the current video snippet
        total_frames = num_frame * config.FRAME_RATE * (1 / 2)

        # Calculating the interval at which information need to be logged
        interval = max(total_frames // num_info_per_chunk, 1)

        # Height and width for cropping the frame
        height, width = config.HEIGHT, config.WIDTH
        x, y = 100, 0

        curr_frame, curr_luv_frame, curr_gray_frame = None, None, None
        prev_frame, prev_luv_frame, prev_gray_frame = None, None, None

        file_paths = []
        frame_diffs = []
        frames = []

        # Capture the video frame by frame
        is_frame, frame = video_capture.read()
        if not is_frame:
            return []
        frame = frame[y:height, x:width, :]  # cropping frames
        copy_frame = frame.copy()  # Frame to be saved
        first_frame = frame.copy()  # To save first frame if no key frames extracted
        frame = cv2.resize(frame, config.dsize)  # resizing the frame
        count = 1

        # Creates an image filled with zero intensities with the same dimensions as the cropped frame
        mask = np.zeros_like(frame)

        # Creates an image filled with zero intensities with the same dimensions as cropped frame
        black_img = np.zeros_like(frame)
        black_image_histogram = self.get_histogram(black_img)

        # Sets image saturation to maximum
        mask[..., 1] = 255

        if config.HASHING is True:
            threshold = config.THRESHOLD["hashing"]
        elif config.DO_OPTICAL_FLOW is True:
            threshold = config.THRESHOLD["optical_flow"]
        else:
            threshold = config.THRESHOLD["absolute_difference"]

        while is_frame:
            if config.CHANGE_FPS:  # changing the FPS
                video_capture.set(
                    cv2.CAP_PROP_POS_MSEC, (count * config.FRAME_RATE * 1000)
                )

            # Conversion to LUV colorspace
            luv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)

            # Conversion to Grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            curr_frame = frame
            curr_luv_frame = luv_frame
            curr_gray_frame = gray_frame
            if curr_frame is not None and prev_frame is not None:
                if config.HASHING is True:
                    # Getting the hash difference (phash, ahash, dhash) for two frames
                    phash, ahash, dhash = utils_tools.hashing_difference(
                        prev_frame, curr_frame
                    )
                    diff_sum = phash + ahash

                elif config.DO_OPTICAL_FLOW:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray_frame, curr_gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    # Computes the magnitude and angle of the 2D vectors
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    # Sets image hue according to the optical flow direction
                    mask[..., 0] = angle * 180 / np.pi / 2

                    # Sets image value according to the optical flow magnitude (normalized)
                    mask[..., 2] = cv2.normalize(
                        magnitude, None, 0, 255, cv2.NORM_MINMAX
                    )

                    # Converts HSV to RGB (BGR) color representation
                    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

                    # Gets histogram of rgb frame
                    curr_histogram = self.get_histogram(rgb)
                    diff_sum = cv2.compareHist(
                        black_image_histogram, curr_histogram, cv2.HISTCMP_BHATTACHARYYA
                    )
                else:
                    # Calculating the absolute difference between two frames
                    frame_diff = cv2.absdiff(curr_luv_frame, prev_luv_frame)
                    diff_sum = np.sum(frame_diff)

                if config.USE_THRESHOLD is True:
                    if diff_sum >= threshold:
                        file_path = self.save_frame(count, copy_frame)
                        file_paths.append(file_path)

                frame_diffs.append(diff_sum)
                frame = Frame(count, copy_frame, diff_sum)
                frames.append(frame)

            if (count + 1) % interval == 0:
                logging.info(f"{count + 1} Frame processed...")

            prev_frame = curr_frame
            prev_luv_frame = curr_luv_frame
            prev_gray_frame = curr_gray_frame
            count = count + 1

            is_frame, frame = video_capture.read()
            if is_frame:
                frame = frame[y:height, x:width, :]
                copy_frame = frame.copy()
                frame = cv2.resize(frame, config.dsize)

            """
            cv2.imshow('frame',luv_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            """

        # Releasing the video capture object after the processing is done
        video_capture.release()

        # Calculating window length for smoothening the array
        window_len = math.floor(math.sqrt(len(frame_diffs))) - 1
        if config.USE_THRESHOLD is False:
            file_paths = self.save_local_maxima(frames, frame_diffs, window_len)

        # If no Keyframes were detected then saving the first frame
        if len(file_paths) == 0:
            file_paths = [self.save_frame(1, first_frame)]

        if chunk_id is None:
            logging.info(f"Evaluation Done...")
        else:
            logging.info(f"Transcript {self.chunk_id} Done...")
        return file_paths
