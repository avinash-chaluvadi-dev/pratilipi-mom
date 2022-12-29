import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    shareMoMButton: {
        color: "#7D7D7D!important",
        borderColor: "#BEBEBE!important",
        textTransform: "none!important",
        width: "220px",
        height: "40px",
        borderRadius: "8px!important",
        border: "solid 2px #f0f5ff!important",
        borderColor: "#f0f5ff!important",
        "&:hover": {
            backgroundColor: "rgba(240, 245, 255, 0.5)!important",
        },
    },
}));

export default useStyles;
