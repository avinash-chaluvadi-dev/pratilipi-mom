import React from "react";
import Dialog from "@material-ui/core/Dialog";
import MuiDialogTitle from "@material-ui/core/DialogTitle";
import MuiDialogContent from "@material-ui/core/DialogContent";
import MuiDialogActions from "@material-ui/core/DialogActions";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import Typography from "@material-ui/core/Typography";
import KeyboardBackspaceIcon from "@material-ui/icons/KeyboardBackspace";
import useStyles from "./styles";

const DialogTitle = (props) => {
    const { children, onClose, onBack, ...other } = props;
    const classes = useStyles();
    return (
        <MuiDialogTitle className={classes.title} {...other}>
            {onBack ? (
                <IconButton
                    aria-label="close"
                    className={classes.backButton}
                    onClick={onBack}
                >
                    <KeyboardBackspaceIcon />
                </IconButton>
            ) : null}
            {onBack ? (
                <Typography className={classes.titleBackbutton}>
                    {children}
                </Typography>
            ) : (
                <Typography className={classes.titleClosebutton}>
                    {children}
                </Typography>
            )}
            {onClose ? (
                <IconButton
                    aria-label="close"
                    className={classes.closeButton}
                    onClick={onClose}
                >
                    <CloseIcon />
                </IconButton>
            ) : null}
        </MuiDialogTitle>
    );
};

const DialogContent = (props) => {
    const { children, ...other } = props;
    const classes = useStyles();
    return (
        <MuiDialogContent
            className={props.isContent ? {} : classes.content}
            {...other}
            style={props.isContent ? { padding: "8px 0" } : {}}
        >
            {children}
        </MuiDialogContent>
    );
};

const DialogActions = (props) => {
    const { children, ...other } = props;
    const classes = useStyles();
    return (
        <MuiDialogActions
            className={classes.action}
            {...other}
            style={props.isContent ? { padding: "5px 8px 8px 8px" } : {}}
        >
            {children}
        </MuiDialogActions>
    );
};

const CustomizedDialogs = (props) => {
    const classes = useStyles();
    const {
        title,
        content,
        actions,
        width,
        open,
        handleClose,
        handleBack,
        isContent,
        isCustomWidth,
        classeNameTitle,
        titleStyle,
    } = props;

    return (
        <div>
            <Dialog
                fullWidth={true}
                maxWidth={width}
                onClose={handleClose}
                open={open}
                className={
                    isCustomWidth ? classes.fullMaxWidth : classes.rounded
                }
            >
                <DialogTitle
                    id="max-width-dialog-title"
                    onClose={handleClose}
                    onBack={handleBack}
                    isContent={isContent}
                    style={titleStyle}
                    className={classeNameTitle}
                >
                    {title}
                </DialogTitle>
                <DialogContent isContent={isContent}>{content}</DialogContent>
                <DialogActions isContent={isContent}>{actions}</DialogActions>
            </Dialog>
        </div>
    );
};

export default CustomizedDialogs;
