import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    uploadFiles: {
        padding: "20px",
        backgroundColor: theme.palette.white.main,
    },
    uploadBox: {
        padding: "20px",
        backgroundColor: theme.palette.white.secondary,
        borderColor: "#8ebbff",
        borderStyle: "dashed",
    },
    table: {
        borderBottom: "none",
    },
    previewTable: {
        borderBottom: "none",
        backgroundColor: theme.palette.grey.main,
    },
    checkBox: {
        backgroundColor: theme.palette.grey.main,
    },
    alignItemsAndJustifyContent: {
        width: "100%",
        height: 350,
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        lineHeight: 2,
    },
    root: {
        "& > * + *": {
            // marginLeft: theme.spacing(1),
        },
    },
}));

export default useStyles;
