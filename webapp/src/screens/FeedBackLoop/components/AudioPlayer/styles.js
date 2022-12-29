import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => {
    return {
        root: {
            backgroundColor: "inherit",
            transition: "inherit",
            margin: "-4px 3px",
            [theme.breakpoints.down("sm")]: {
                width: "100%",
                backgroundColor: "inherit",
                transition: "inherit",
            },
        },
        icon: {
            fontSize: "20px",
        },
        loopIcon: {
            color: "#3f51b5",
            "&.selected": {
                color: "#0921a9",
            },
            "&:hover": {
                color: "#7986cb",
            },
            [theme.breakpoints.down("sm")]: {
                display: "none",
            },
        },
        playIcon: {
            width: "20px",
            height: "20px",
            color: "gray",
            "&:hover": {
                color: "primary",
            },
        },
        pauseIcon: {
            width: "20px",
            height: "20px",
            color: "gray",
            "&:hover": {
                color: "primary",
            },
        },
        volumeIcon: {
            color: "rgba(0, 0, 0, 0.54)",
        },
        volumeSlider: {
            color: "black",
        },
        progressTime: {
            color: "rgba(0, 0, 0, 0.54)",
        },
        mainSlider: {
            color: "#3f51b5",
            margin: "0px 0 0 14px",
            width: "95%",
            "& .MuiSlider-rail": {
                color: "#7986cb",
            },
            "& .MuiSlider-track": {
                color: "#3f51b5",
            },
            "& .MuiSlider-thumb": {
                color: "#303f9f",
            },
        },
    };
});

export default useStyles;
