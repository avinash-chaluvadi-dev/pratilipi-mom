import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    datePickerGrid: {
        // width: "15%",
        margin: "1px 13px 10px 3px",
        border: "0.8px solid #EAEAEA",
        color: "#645BF5",
        "&:hover": {
            borderColor: "#EAEAEA",
            background: "#EAEAEA",
            cursor: "pointer",
        },
        "& .MuiIconButton-root": {
            padding: "0px",
            margin: "0px",
        },
        "& .MuiSvgIcon-root": {
            width: "0.7em",
            height: "0.7em",
        },
        "& .MuiInputBase-input": {
            padding: "0px",
            margin: "0px",
            minWidth: "10px",
            maxWidth: "68px",
            color: "#645BF5",
            // fontSize:'12px',
        },
        "& .MuiInputAdornment-positionStart": {
            margin: "0px",
        },
        "& .MuiInputAdornment-positionEnd": {
            margin: "0px",
        },
        "& .MuiInput-underline:before": {
            content: "none!important",
            borderBottom: "0px",
        },
        "& .MuiInput-underline:after": {
            content: "none!important",
            borderBottom: "0px",
        },
    },
    calenderIcon: {
        color: "#645BF5",
    },
    paperModalPopup: {
        padding: theme.spacing(2),
        margin: "0px 0px 0px 15px",
        textAlign: "center",
        color: theme.palette.text.secondary,
        // border: '1px solid #CFCFCF',
        height: "50px",
        background: "#F8F8F8",
        width: "100%",
        boxShadow: "none",
    },
    modalWidtHeight: {
        "& .MuiDialog-paperWidthMd": {
            width: "838px",
            maxWidth: "838px",
            height: "504px",
        },
        "& .MuiDialog-paperFullWidth": {
            width: "838px",
            maxWidth: "838px",
            height: "504px",
        },
    },
    modalTitleBar: {
        borderBottom: "1px solid #D6D6D6",
        margin: "20px 0px 0px 0px",
    },
    modalContentBar: {
        borderBottom: "1px solid #D6D6D6",
        margin: "0px 0px 20px 0px",
    },
    modalContentTitleBar: {
        borderBottom: "1px solid #D6D6D6",
        margin: "20px 0px 20px 0px",
    },
    textOverflow: {
        // maxWidth: '90%',
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        // top: 40%
    },
}));

export default useStyles;
