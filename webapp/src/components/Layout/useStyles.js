import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    mainContent: {
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "flex-start",
        marginTop: "80px",
        marginLeft: "100px",
        marginRight: "10px",
        width: `calc(100% - ${theme.spacing(20) + 1}px)`,
        transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    },
    mainContentShift: {
        marginLeft: "250px",
        backgroundColor: theme.palette.white.secondary,
        width: `calc(100% - 300px)`,
        transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    },
}));

export default useStyles;
