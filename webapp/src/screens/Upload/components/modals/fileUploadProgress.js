import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";
import LinearProgress, {
    linearProgressClasses,
} from "@mui/material/LinearProgress";
import { styled } from "@mui/material/styles";
import Box from "@material-ui/core/Box";
import Modal from "components/Modal";
import Button from "@material-ui/core/Button";
import inprogress from "static/images/inprogress.png";
import useStyles from "../../useStyles";
import CheckCircleIcon from "@material-ui/icons/CheckCircle";
import FileUploadCancellation from "./fileUploadCancellation";
import PreviewFile from "./previewFile";
import { connect, useDispatch } from "react-redux";

const BorderLinearProgress = styled(LinearProgress)(({ theme }) => ({
    height: 10,
    borderRadius: 5,
    [`&.${linearProgressClasses.colorPrimary}`]: {
        backgroundColor:
            theme.palette.grey[theme.palette.mode === "light" ? 200 : 800],
    },
    [`& .${linearProgressClasses.bar}`]: {
        borderRadius: 5,
        backgroundColor: theme.palette.mode === "light" ? "#1a90ff" : "#308fe8",
    },
}));

function LinearProgressWithLabel(props) {
    return (
        <Box display="flex" alignItems="center">
            <Box width="50%" ml={25}>
                <BorderLinearProgress variant="determinate" {...props} />
            </Box>
            <Box>
                {props.value === 100 ? (
                    <Box color="green" ml={2}>
                        <CheckCircleIcon style={{ fill: "green" }} />
                        {`${Math.round(props.value)}%`}
                    </Box>
                ) : (
                    <Box ml={2}>{`${Math.round(props.value)}%`}</Box>
                )}
            </Box>
        </Box>
    );
}

LinearProgressWithLabel.propTypes = {
    value: PropTypes.number.isRequired,
};

const FileUploadProgress = ({
    openFileUploadProgress,
    handleFileUploadProgressModalProceed,
    handleFileUploadProgressClose,
    handlePayRecording,
    fileName,
    fileSize,
    progress,
    uploadReducer,
    transcribeFileUrl,
}) => {
    const classes = useStyles();
    const dispatch = useDispatch();
    const [
        openConfirmUploadCancellationModal,
        setOpenConfirmUploadCancellationModal,
    ] = useState(false);
    const [openPreview, setOpenPreview] = useState(false);
    const [disabled, setDisabled] = useState(true);

    useEffect(() => {
        if (progress === 100 && transcribeFileUrl) {
            setDisabled(false);
        }
    }, [dispatch, progress, transcribeFileUrl]);

    const preview = openPreview ? (
        <PreviewFile
            openPreview={openPreview}
            handlePreviewClose={() => setOpenPreview(false)}
            fileName={fileName}
            fileSize={fileSize}
            src={uploadReducer.playbackFileUrl}
            handleCancellation={handleFileUploadProgressClose}
        />
    ) : null;

    const title = (
        <Box
            style={{
                fontSize: "20px",
                fontWeight: "bold",
                color: "#333333",
            }}
        >
            In progress
        </Box>
    );
    const content = (
        <Box className={classes.alignItemsAndJustifyContent}>
            <img src={inprogress} alt="Upload" />
            <LinearProgressWithLabel value={progress} />
            {progress !== 100 ? (
                <Box
                    style={{
                        fontSize: "20px",
                        fontWeight: "normal",
                        color: "#333333",
                        lineHeight: 4,
                        fontFamily: "Lato",
                    }}
                >
                    Recording uploading is in progress. Please wait and do not
                    close it.
                </Box>
            ) : (
                <Box
                    style={{
                        fontSize: "20px",
                        fontWeight: "normal",
                        color: "#333333",
                        lineHeight: 4,
                        fontFamily: "Lato",
                    }}
                >
                    Recording is uploaded successfully. Play it if you want to
                    confirm once or you can proceed.
                </Box>
            )}
        </Box>
    );
    const actions = (
        <Box className={classes.root}>
            <Button
                autoFocus
                onClick={() => {
                    setOpenPreview(true);
                    handlePayRecording();
                }}
                variant="contained"
                style={{
                    textTransform: "none",
                    maxWidth: "400px",
                    maxHeight: "40px",
                    minWidth: "175px",
                    minHeight: "40px",
                    fontSize: "16px",
                    fontWeight: "bold",
                    fontFamily: "Lato",
                    color: "#286ce2",
                    backgroundColor: "#FFFFFF",
                    borderRadius: "8px",
                    border: "2px solid rgb(240, 245, 255)",
                }}
                disabled={disabled}
            >
                Play Recording
            </Button>
            <Button
                autoFocus
                onClick={handleFileUploadProgressModalProceed}
                variant="contained"
                style={{
                    textTransform: "none",
                    maxWidth: "400px",
                    maxHeight: "40px",
                    minWidth: "175px",
                    minHeight: "40px",
                    fontSize: "16px",
                    fontWeight: "bold",
                    fontFamily: "Lato",
                    color: "#FFFFFF",
                    backgroundColor: disabled ? "#9bbaeb" : "#1665DF",
                    borderRadius: "8px",
                }}
                disabled={disabled}
            >
                Proceed
            </Button>
            {preview}
        </Box>
    );

    return (
        <Box>
            <Modal
                title={title}
                content={content}
                actions={actions}
                width="md"
                open={openFileUploadProgress}
                handleClose={() => {
                    setOpenConfirmUploadCancellationModal(
                        !openConfirmUploadCancellationModal
                    );
                }}
            />
            {openConfirmUploadCancellationModal ? (
                <FileUploadCancellation
                    openConfirmUploadCancellation={
                        openConfirmUploadCancellationModal
                    }
                    handleConfirmUploadCancellationClose={() => {
                        setOpenConfirmUploadCancellationModal(
                            !openConfirmUploadCancellationModal
                        );
                    }}
                    handleFileUploadProgressClose={
                        handleFileUploadProgressClose
                    }
                />
            ) : null}
        </Box>
    );
};

export default connect(
    ({ uploadReducer }) => ({ uploadReducer }),
    {}
)(FileUploadProgress);
