import React, { useState, useRef, useEffect } from "react";
import Button from "@material-ui/core/Button";
import play from "static/images/play.png";
import pause from "static/images/pause.png";
import forward from "static/images/forward.png";
import backward from "static/images/backward.png";

const AudioPlayer = (meeting) => {
    // state
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);

    // references
    const audioPlayer = useRef(); // reference our audio component
    const progressBar = useRef(); // reference our progress bar
    const animationRef = useRef(); // reference the animation

    useEffect(() => {
        const seconds = Math.floor(audioPlayer.current.duration);
        setDuration(seconds);
        progressBar.current.max = seconds;
    }, [
        audioPlayer?.current?.loadedmetadata,
        audioPlayer?.current?.readyState,
    ]);

    const calculateTime = (secs) => {
        const minutes = Math.floor(secs / 60);
        const returnedMinutes = minutes < 10 ? `0${minutes}` : `${minutes}`;
        const seconds = Math.floor(secs % 60);
        const returnedSeconds = seconds < 10 ? `0${seconds}` : `${seconds}`;
        return `${returnedMinutes}:${returnedSeconds}`;
    };

    const togglePlayPause = () => {
        const prevValue = isPlaying;
        setIsPlaying(!prevValue);
        if (!prevValue) {
            audioPlayer.current.play();
            animationRef.current = requestAnimationFrame(whilePlaying);
        } else {
            audioPlayer.current.pause();
            cancelAnimationFrame(animationRef.current);
        }
    };

    const whilePlaying = () => {
        progressBar.current.value = audioPlayer.current.currentTime;
        changePlayerCurrentTime();
        animationRef.current = requestAnimationFrame(whilePlaying);
    };

    const changeRange = () => {
        audioPlayer.current.currentTime = progressBar.current.value;
        changePlayerCurrentTime();
    };

    const changePlayerCurrentTime = () => {
        progressBar.current.style.setProperty(
            "--seek-before-width",
            `${(progressBar.current.value / duration) * 100}%`
        );
        setCurrentTime(progressBar.current.value);
    };

    const backTen = () => {
        progressBar.current.value = Number(progressBar.current.value - 10);
        changeRange();
    };

    const forwardTen = () => {
        progressBar.current.value = Number(progressBar.current.value + 10);
        changeRange();
    };

    return (
        <div>
            <audio ref={audioPlayer} src={meeting} preload="metadata"></audio>
            <Button onClick={backTen}>
                <img src={backward} alt="30 seconds backward" />
            </Button>
            <Button onClick={togglePlayPause}>
                {isPlaying ? (
                    <img src={pause} alt="Pause" />
                ) : (
                    <img src={play} alt="PLay" />
                )}
            </Button>
            <Button onClick={forwardTen}>
                <img src={forward} alt="30 seconds forward" />
            </Button>

            {/* current time */}
            <div>{calculateTime(currentTime)}</div>

            {/* progress bar */}
            <div>
                <input
                    type="range"
                    defaultValue="0"
                    ref={progressBar}
                    onChange={changeRange}
                />
            </div>

            {/* duration */}
            <div>{duration && !isNaN(duration) && calculateTime(duration)}</div>
        </div>
    );
};

export default AudioPlayer;
