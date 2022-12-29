import { makeStyles } from "@material-ui/core/styles";

const customTableStyles = makeStyles((theme) => ({
    cellStyle: {
        borderRight: "1px solid #eee !important",
        color: "primary!important",
        fontSize: "16px!important",
        cursor: "pointer!important",
        borderBottom: "1px solid #eee !important",
        borderLeft: "1px solid #eee !important",
        padding: "5px 20px!important",
        color: "#333!important",
        fontFamily: "Lato!important",
    },
    headerStyle: {
        borderRight: "1px solid #eee!important",
        textAlign: "left!important",
        color: "#666!important",
        fontFamily: "Lato!important",
        fontSize: "14px!important",
        fontWeight: "bold!important",
        cursor: "pointer!important",
        borderBottom: "0px!important",
        padding: "5px 20px!important",
    },
    root: {
        width: "100%",
        "&:nth-of-type(odd)": {
            backgroundColor: "white",
            background: "white",
        },
        "&:nth-of-type(even)": {
            backgroundColor: "grey",
            background: "grey",
        },
    },
    container: {
        maxHeight: 440,
        borderRadius: "8px",
    },
}));

export default customTableStyles;
