import React, { useState } from "react";
import Modal from "components/Modal";
import Box from "@material-ui/core/Box";
import MediaPlayer from "components/MediaPlayer";
import Button from "@material-ui/core/Button";
import FileUploadCancellation from "./fileUploadCancellation";

const PreviewFile = ({
    openPreview,
    handlePreviewClose,
    fileName,
    fileSize,
    src,
    handleCancellation,
}) => {
    const fileExtension = fileName.substr(fileName.lastIndexOf("."));

    const [
        openConfirmUploadCancellationModal,
        setOpenConfirmUploadCancellationModal,
    ] = useState(false);

    const title = (
        <Box
            style={{
                color: "#333333",
                fontSize: "20px",
                fontWeight: "bold",
            }}
        >
            Play Recording
        </Box>
    );

    const content = (
        <Box>
            <Box>
                <MediaPlayer fileExtension={fileExtension} meeting={src} />
            </Box>
        </Box>
    );
    const actions = handleCancellation ? (
        <>
            <Button
                onClick={() => {
                    setOpenConfirmUploadCancellationModal(
                        !openConfirmUploadCancellationModal
                    );
                }}
                variant="contained"
                color="primary"
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
            >
                Cancel MoM Generation
            </Button>
            <Button
                autoFocus
                onClick={handlePreviewClose}
                variant="contained"
                color="primary"
                style={{
                    textTransform: "none",
                    maxWidth: "400px",
                    maxHeight: "40px",
                    minWidth: "175px",
                    minHeight: "40px",
                    fontSize: "16px",
                    color: "#FFFFFF",
                    backgroundColor: "#1665DF",
                    borderRadius: "8px",
                }}
            >
                Continue
            </Button>
        </>
    ) : null;

    return (
        <Box>
            <Modal
                title={title}
                content={content}
                actions={actions}
                width="md"
                open={openPreview}
                handleBack={handlePreviewClose}
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
                    handleFileUploadProgressClose={handleCancellation}
                />
            ) : null}
        </Box>
    );
};

export default PreviewFile;
