import React from "react";
import Backdrop from "@material-ui/core/Backdrop";
import Loader from "react-loader-spinner";
import { makeStyles } from "@material-ui/core/styles";
import { useDispatch, useSelector } from "react-redux";

const useStyles = makeStyles((theme) => ({
    backdrop: {
        zIndex: theme.zIndex.drawer + 1,
        color: "#fff",
        backgroundColor: "rgba(0, 0, 0, 0.2)",
    },
}));

const LoaderComponent = () => {
    const classes = useStyles();
    const dispatch = useDispatch();
    const { isLoader } = useSelector((state) => state.loaderReducer);
    const handleClose = () => {
        dispatch({
            type: "STOP_LOADER",
            payload: { isLoader: false },
        });
    };

    return (
        <div>
            <Backdrop
                className={classes.backdrop}
                open={isLoader}
                onClick={handleClose}
            >
                <Loader
                    type="ThreeDots"
                    color="#1464DF"
                    height={100}
                    width={100}
                />
            </Backdrop>
        </div>
    );
};

export default LoaderComponent;
