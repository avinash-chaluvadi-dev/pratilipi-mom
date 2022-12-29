import Modal from "components/Modal";
import Box from "@material-ui/core/Box";
import MediaPlayer from "components/MediaPlayer";

const PreviewFile = ({
    openPreview,
    handlePreviewClose,
    fileName,
    fileSize,
    src,
}) => {
    const fileExtension = fileName.substr(fileName.lastIndexOf("."));

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
            <MediaPlayer fileExtension={fileExtension} meeting={src} />
        </Box>
    );
    const actions = "";

    return (
        <Modal
            title={title}
            content={content}
            actions={actions}
            width="md"
            open={openPreview}
            // handleClose={handlePreviewClose}
            handleBack={handlePreviewClose}
        />
    );
};

export default PreviewFile;
