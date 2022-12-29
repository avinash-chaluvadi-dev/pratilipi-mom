import Button from "@material-ui/core/Button";
import Modal from "components/Modal";
import Box from "@material-ui/core/Box";
import useStyles from "../../useStyles";

const InvalidFile = ({
    openInvalidFileModal,
    handleInvalidFileModalClose,
    handleInputChangeonReUpload,
    titleContent,
    bodyContent,
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
            {titleContent}
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
            {bodyContent}
        </Box>
    );
    const actions = (
        <Box className={classes.root}>
            <Button
                autoFocus
                onClick={handleInvalidFileModalClose}
                variant="contained"
                style={{
                    textTransform: "none",
                    maxWidth: "400px",
                    maxHeight: "40px",
                    minWidth: "175px",
                    minHeight: "40px",
                    fontSize: "16px",
                    fontWeight: "bold",
                    color: "#286ce2",
                    backgroundColor: "#FFFFFF",
                    borderRadius: "8px",
                    border: "2px solid rgb(240, 245, 255)",
                    fontFamily: "Lato",
                }}
            >
                Cancel
            </Button>
            <Button
                variant="contained"
                component="label"
                style={{
                    textTransform: "none",
                    maxWidth: "400px",
                    maxHeight: "40px",
                    minWidth: "175px",
                    minHeight: "40px",
                    fontSize: "16px",
                    fontWeight: "bold",
                    color: "#FFFFFF",
                    backgroundColor: "#1665DF",
                    borderRadius: "8px",
                    fontFamily: "Lato",
                }}
            >
                Upload New Recording
                <input
                    type="file"
                    hidden
                    onChange={handleInputChangeonReUpload}
                />
            </Button>
        </Box>
    );

    return (
        <Modal
            title={title}
            content={content}
            actions={actions}
            width="sm"
            open={openInvalidFileModal}
            handleClose={handleInvalidFileModalClose}
            classeNameTitle={classes.modalTitle}
        />
    );
};

export default InvalidFile;
