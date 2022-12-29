import Button from "@material-ui/core/Button";
import Modal from "components/Modal";
import Box from "@material-ui/core/Box";
import useStyles from "../../useStyles";

const FileUploadCancellation = ({
    openConfirmUploadCancellation,
    handleConfirmUploadCancellationClose,
    handleFileUploadProgressClose,
}) => {
    const classes = useStyles();
    const title = (
        <Box
            style={{
                fontSize: "24px",
                fontWeight: "medium",
                textAlign: "left",
                color: "#333333",
            }}
        >
            Are you sure you want to cancel?
        </Box>
    );
    const content = (
        <Box
            style={{
                fontSize: "14px",
                textAlign: "left",
                color: "#333333",
            }}
        >
            You cannot reverse this action once you close it. Make sure you want
            to close it and do not want it back again.
        </Box>
    );
    const actions = (
        <Box className={classes.root}>
            <Button
                autoFocus
                onClick={handleConfirmUploadCancellationClose}
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
            >
                No, Continue
            </Button>
            <Button
                autoFocus
                onClick={handleFileUploadProgressClose}
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
                    color: "#FFFFFF",
                    backgroundColor: "#1665DF",
                    borderRadius: "8px",
                }}
            >
                Yes, Cancel
            </Button>
        </Box>
    );

    return (
        <Modal
            title={title}
            content={content}
            actions={actions}
            width="sm"
            open={openConfirmUploadCancellation}
            handleClose={handleConfirmUploadCancellationClose}
            classeNameTitle={classes.modalTitle}
        />
    );
};

export default FileUploadCancellation;
