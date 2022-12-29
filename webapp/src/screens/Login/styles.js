import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    loginForm: {
        width: "300px",
        "& .MuiOutlinedInput-root": {
            borderRadius: "8px",
            border: "solid 0.8px #949494",
        },
        "& .MuiOutlinedInput-input": {
            fontSize: "14px",
            color: "#666666",
        },
    },
    headerText: {
        fontSize: "24px",
        fontFamily: "Bitter",
        fontWeight: "medium",
        textAlign: "left",
    },
    bodyText: {
        fontSize: "14px",
        fontWeight: "bold",
        lineHeight: "40px",
        textAlign: "left",
    },
    button: {
        textTransform: "none",
        width: "300px",
        height: "40px",
        margin: "20px 0 0px 0px",
        borderRadius: "8px",
        fontSize: "16px",
        fontFamily: "Lato",
        fontWeight: "bold",
        textAlign: "center",
    },
    outerGrid: {
        minHeight: "90vh",
        width: "100%",
        overflow: "hidden",
        alignItems: "center",
        justifyContent: "center",
    },
    divider: {
        marginTop: "5vh",
        height: "30vh",
    },
    outerBox: {
        display: "flex",
        alignItems: "center",
        width: "fit-content",
    },
    logoBox: {
        marginTop: "40px",
        marginRight: "100px",
        display: "grid",
    },
    errorText: {
        fontSize: "12px",
        color: "#d20a3c",
        alignItems: "center",
        display: "flex",
    },
    momSummary: {
        marginTop: "25px",
        fontSize: "14.7px",
        fontWeight: "bold",
        opacity: "0.8",
        lineHeight: "20px",
        maxWidth: "470px",
        textAlign: "justify",
    },
    momLogo: {
        height: "auto",
        width: "300px",
        marginLeft: "auto",
    },
}));

export default useStyles;
