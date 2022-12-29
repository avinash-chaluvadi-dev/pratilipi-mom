import React, { useEffect, useState } from "react";
import Button from "@material-ui/core/Button";
import Box from "@material-ui/core/Box";
import Card from "@material-ui/core/Card";
import upload from "static/images/uploadimg.png";
import useStyles from "../../useStyles";
import { Grid } from "@material-ui/core";
import FileConfirmation from "../modals/fileConfirmation";
import InvalidFile from "../modals/invalidFile";
import FileUploadProgress from "../modals/fileUploadProgress";
import ErrorOutlineOutlinedIcon from "@material-ui/icons/ErrorOutlineOutlined";
import {
    uploadFile,
    cancelFileUpload,
    getUploadFileTrascription,
    getPlayBackFilePath,
    getAllUploadedFiles,
} from "store/action/upload";
import { useSelector, useDispatch } from "react-redux";
import axios from "axios";

const Upload = () => {
    const classes = useStyles();
    const dispatch = useDispatch();
    const [file, setFile] = useState("");
    const [teamName, setTeamName] = useState("");
    const [transcribeFileUrl, setTranscribeFileUrl] = useState("");

    const [openInvalidFileModal, setOpenInvalidFileModal] = useState(false);
    const [openInvalidFileSizeModal, setOpenInvalidFileSizeModal] =
        useState(false);
    const [showError, setShowError] = useState(false);
    const [showInvalidFileExtensionError, setShowInvalidFileExtensionError] =
        useState(false);
    const [showInvalidFileSizeError, setShowInvalidFileSizeError] =
        useState(false);
    const [openFileConfirmationModal, setOpenFileConfirmationModal] =
        useState(false);
    const [openFileUploadProgressModal, setOpenFileUploadProgressModal] =
        useState(false);
    const [progress, setProgress] = useState(0);
    const [source, setSource] = useState(axios.CancelToken.source());
    const [isCancelled, setIsCancelled] = useState(false);
    const { currentUploadedFile } = useSelector((state) => state.uploadReducer);

    useEffect(() => {
        if (currentUploadedFile?.file) {
            const filePath = currentUploadedFile.file;
            const fileUrlArray = filePath.split(":");
            const transcribeUrlData = fileUrlArray.splice(1, 2);
            const combineUrl = transcribeUrlData.join(":");
            const finalUrl = combineUrl.substring(2);
            setTranscribeFileUrl(finalUrl);
        }
        if (isCancelled) setSource(axios.CancelToken.source());

        return () => setTranscribeFileUrl("");
    }, [isCancelled, currentUploadedFile?.file]);

    function fileExtensionAndSizeCheck(selectedFile) {
        let selectedFileName = selectedFile.name;
        let selectedFileExtension = selectedFileName.substr(
            selectedFileName.lastIndexOf(".")
        );
        let selectedFileSize = selectedFile.size;
        if (selectedFileSize > 41943040) {
            setOpenInvalidFileSizeModal(true);
        } else if (
            selectedFileExtension === ".mp4" ||
            selectedFileExtension === ".mp3" ||
            selectedFileExtension === ".wav"
        ) {
            setFile(selectedFile);
            setOpenFileConfirmationModal(true);
            setShowInvalidFileExtensionError(false);
            setShowInvalidFileSizeError(false);
            setShowError(false);
        } else {
            setOpenInvalidFileModal(true);
        }
    }

    const handleDrop = (e) => {
        e.preventDefault();
        let files = [...e.dataTransfer.files];
        fileExtensionAndSizeCheck(files[0]);
    };

    return (
        <Grid xs={12}>
            <Card
                className={
                    showError ? classes.uploadFilesError : classes.uploadFiles
                }
            >
                <Box
                    style={{
                        fontSize: "20px",
                        fontWeight: "bold",
                        textAlign: "left",
                        color: "#464546",
                    }}
                >
                    Upload Recording
                </Box>
                <Box
                    style={{
                        fontSize: "14px",
                        textAlign: "left",
                        color: "#666666",
                        lineHeight: 3,
                    }}
                >
                    Supported formats are:{" "}
                    <Box
                        style={{
                            fontSize: "14px",
                            fontWeight: "bold",
                            textAlign: "left",
                            color: "#464546",
                            display: "inline",
                        }}
                    >
                        .mp4, .mp3 and .wav
                    </Box>
                </Box>
                <Box
                    onDrop={(e) => handleDrop(e)}
                    onDragOver={(e) => e.preventDefault()}
                    onDragEnter={(e) => e.preventDefault()}
                    onDragLeave={(e) => e.preventDefault()}
                    className={classes.uploadBox}
                >
                    <img src={upload} alt="Upload" />
                    <Box
                        style={{
                            fontSize: "18px",
                            fontWeight: "bold",
                            textAlign: "center",
                            lineHeight: 2,
                        }}
                    >
                        Upload Audio/Video Recording Here
                    </Box>
                    <Box
                        style={{
                            fontSize: "14px",
                            textAlign: "center",
                            color: "#666666",
                            lineHeight: 2,
                        }}
                    >
                        Drag and Drop Recording Here
                    </Box>
                    <Box lineHeight={4}>
                        <Button
                            variant="contained"
                            size="medium"
                            color="primary"
                            component="label"
                            style={{
                                textTransform: "none",
                                maxWidth: "400px",
                                maxHeight: "40px",
                                fontSize: "16px",
                                fontWeight: "bold",
                                fontFamily: "Lato",
                                borderRadius: "8px",
                            }}
                        >
                            Select Recording to Upload
                            <input
                                type="file"
                                value=""
                                hidden
                                onChange={(e) => {
                                    fileExtensionAndSizeCheck(
                                        e.target.files[0]
                                    );
                                }}
                            />
                        </Button>
                    </Box>
                    <Box
                        style={{
                            fontSize: "14px",
                            textAlign: "center",
                            color: "#666666",
                            lineHeight: 2,
                        }}
                    >
                        *Upload recording in supported formats upto 40 MB
                    </Box>
                </Box>
                {showInvalidFileExtensionError ? (
                    <Box
                        style={{
                            fontSize: "12px",
                            textAlign: "left",
                            color: "#FF0000",
                        }}
                    >
                        <ErrorOutlineOutlinedIcon
                            style={{
                                fill: "red",
                                textAlign: "center",
                                verticalAlign: "text-bottom",
                            }}
                        />{" "}
                        Recording format is not supported. Please check the
                        format once and try again. Supported fomats are .mp4 ,
                        .mp3 and .wav.
                    </Box>
                ) : null}
                {showInvalidFileSizeError ? (
                    <Box
                        style={{
                            fontSize: "12px",
                            textAlign: "left",
                            color: "#FF0000",
                        }}
                    >
                        <ErrorOutlineOutlinedIcon
                            style={{
                                fill: "red",
                                textAlign: "center",
                                verticalAlign: "text-bottom",
                            }}
                        />{" "}
                        Recording size exceeded the limit of 40MB. Please try
                        with another file of size 40MB or lower.
                    </Box>
                ) : null}
            </Card>
            {openFileConfirmationModal ? (
                <FileConfirmation
                    openFileConfirmation={openFileConfirmationModal}
                    handleFileConfirmationClose={() => {
                        setOpenFileConfirmationModal(false);
                        setTeamName("");
                    }}
                    file={file}
                    teamName={teamName}
                    setTeamName={setTeamName}
                    submitAction={() => {
                        setOpenFileConfirmationModal(false);
                        setOpenFileUploadProgressModal(true);
                        setTranscribeFileUrl("");
                        let formData = new FormData();
                        formData.append("file", file);
                        formData.append("team_name", teamName);
                        formData.append("file_size", file.size);
                        setProgress(0);
                        dispatch(uploadFile(formData, setProgress, source));
                    }}
                />
            ) : null}
            {openInvalidFileModal ? (
                <InvalidFile
                    openInvalidFileModal={openInvalidFileModal}
                    handleInvalidFileModalClose={() => {
                        setOpenInvalidFileModal(false);
                        setShowInvalidFileSizeError(false);
                        setShowError(true);
                        setShowInvalidFileExtensionError(true);
                    }}
                    handleInputChangeonReUpload={(e) => {
                        setOpenInvalidFileModal(false);
                        fileExtensionAndSizeCheck(e.target.files[0]);
                    }}
                    titleContent="Recording format is not supported!"
                    bodyContent="The recording you are trying to upload is not in a valid format. Please upload
          valid formats like .mp4, .mp3 and .wav"
                />
            ) : null}
            {openInvalidFileSizeModal ? (
                <InvalidFile
                    openInvalidFileModal={openInvalidFileSizeModal}
                    handleInvalidFileModalClose={() => {
                        setOpenInvalidFileSizeModal(false);
                        setShowInvalidFileExtensionError(false);
                        setShowError(true);
                        setShowInvalidFileSizeError(true);
                    }}
                    handleInputChangeonReUpload={(e) => {
                        setOpenInvalidFileSizeModal(false);
                        fileExtensionAndSizeCheck(e.target.files[0]);
                    }}
                    titleContent="Exceeded Size Limit!"
                    bodyContent="Please try to upload a recording having size of 40MB and below."
                />
            ) : null}
            {openFileUploadProgressModal ? (
                <FileUploadProgress
                    openFileUploadProgress={openFileUploadProgressModal}
                    handleFileUploadProgressModalProceed={() => {
                        setOpenFileUploadProgressModal(false);
                        if (transcribeFileUrl) {
                            dispatch(
                                getUploadFileTrascription(transcribeFileUrl)
                            );
                        }
                        setTeamName("");
                    }}
                    handleFileUploadProgressClose={() => {
                        if (progress !== 100) {
                            source.cancel("Operation canceled by the user.");
                            setIsCancelled(true);
                        } else {
                            dispatch(
                                cancelFileUpload(
                                    currentUploadedFile.masked_request_id
                                )
                            );
                            dispatch(getAllUploadedFiles());
                        }
                        setOpenFileUploadProgressModal(false);
                        setFile("");
                    }}
                    fileName={file.name}
                    fileSize={file.size}
                    progress={progress}
                    handlePayRecording={() => {
                        dispatch(getPlayBackFilePath(transcribeFileUrl));
                    }}
                    transcribeFileUrl={transcribeFileUrl}
                />
            ) : null}
        </Grid>
    );
};

export default Upload;
