import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    root: {
        padding: "20px",
    },
    title: {
        padding: "20px",
        paddingTop: "30px",
    },
    content: {
        padding: "20px",
        fontFamily: "Lato",
        fontWeight: "normal",
        fontStretch: "normal",
        fontStyle: "normal",
        lineHeight: "1.25",
        letterSpacing: "normal",
        // color: '#333',
    },
    action: {
        padding: "20px",
    },

    backButton: {
        position: "absolute",
        left: theme.spacing(1),
        top: theme.spacing(1),
        color: "#333333",
    },
    closeButton: {
        position: "absolute",
        right: theme.spacing(1),
        top: theme.spacing(1),
        color: theme.palette.grey[500],
    },
    titleBackbutton: {
        position: "absolute",
        left: theme.spacing(6),
        right: theme.spacing(2),
        top: theme.spacing(2),
    },
    titleClosebutton: {
        position: "absolute",
        left: theme.spacing(2),
        right: theme.spacing(6),
        top: theme.spacing(2),
    },
    titlebold: {
        fontFamily: "Bitter !important",
        fontSize: "24px",
        fontWeight: "bold",
        fontStretch: "normal",
        fontStyle: "normal",
        color: "#333",
    },
    rounded: {
        "& .MuiPaper-rounded": {
            borderRadius: "16px",
            overflowY: "inherit",
        },
    },
    fullMaxWidth: {
        "& .MuiDialog-paperWidthMd": {
            maxWidth: "1376px",
            minHeight: "685px",
        },
        "& .MuiPaper-rounded": {
            borderRadius: "16px",
        },
    },
}));

export default useStyles;
