import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    Input: {
        width: "300px",
        height: "35px",
        margin: "4px 0 0",
        padding: "9px 47px 2px 12px",
        borderRadius: "8px",
        boxShadow: "0 2px 4px 0 rgba(0, 0, 0, 0.08)",
        border: "solid 1px #949494",
        backgroundColor: "#fff",
        "&::placeholder": {
            fontFamily: "Lato",
            fontSize: "14px",
        },
    },
    InputLabel: {
        width: "100%",
        height: "22px",
        marginBottom: "4px",
        fontSize: "13px",
        fontWeight: "bold",
        lineHeight: "1.57",
        fontFamily: "lato",
        color: theme.palette.dark.main,
    },
    HelperText: {
        width: "175px",
        height: "13px",
        margin: "5px 0 0 7.8px",
        fontFamily: "Lato",
        fontSize: "12px",
        color: "#d20a3c",
    },
    ErrorBorder: {
        border: "1px solid #d20a3c",
    },
}));

export default useStyles;
