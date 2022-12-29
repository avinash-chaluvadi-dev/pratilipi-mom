import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    uploadFiles: {
        padding: "25px",
        backgroundColor: theme.palette.white.main,
        borderRadius: "16px",
        boxShadow: "0px 8px 16px 0px rgba(0, 0, 0, 0.2)",
        border: "1px solid rgba(0, 0, 0, 0.08)",
        lineHeight: 1,
    },
    uploadFilesError: {
        padding: "25px",
        paddingBottom: "10px",
        backgroundColor: theme.palette.white.main,
        borderRadius: "16px",
        boxShadow: "0px 8px 16px 0px rgba(0, 0, 0, 0.2)",
        border: "1px solid rgba(0, 0, 0, 0.08)",
        lineHeight: 1,
    },
    uploadBox: {
        padding: "30px",
        borderColor: "#a3a3a3",
        border: "2px dashed",
        borderRadius: "16px",
    },
    table: {
        borderBottom: "none",
    },
    rowHover: {
        "&:hover": {
            backgroundColor: "rgba(255, 255, 255, 0.13) !important",
        },
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
        textAlign: "center",
    },
    formControl: {
        minWidth: 100,
    },
    select: {
        fontSize: "14px",
        fontWeight: "bold",
        fontFamily: "Lato",
        color: "#286ce2",
        paddingRight: "5px",
        borderRadius: "8px",
        outline: "1px solid #949494",
        boxShadow: "0px 1px 2px 0px rgba(0, 0, 0, 0.08)",
        height: "40px",
        width: "180px",
    },
    menu: {
        fontSize: "14px",
        fontFamily: "lato",
        fontWeight: "bold",
        width: "180px",
        color: "#333333",
        "&:hover": {
            color: "#ffffff",
            backgroundColor: "#286ce2",
        },
    },
    modalTitle: {
        borderRadius: "16px",
        height: "50px",
        background: "#f7f7f7 padding-box",
    },
    root: {
        "& > * + *": {
            marginLeft: theme.spacing(1),
        },
    },
}));

export default useStyles;
