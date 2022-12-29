import React from "react";
import { Typography, Box } from "@material-ui/core";
import useStyles from "../useStyles";
import SearchImg from "static/Icons/search.png";

const NoData = () => {
    const classes = useStyles();
    return (
        <Box className={classes.ActualContent}>
            <img
                src={SearchImg}
                alt=""
                style={{ width: "146px", height: "146px" }}
            ></img>
            <Typography
                style={{
                    fontSize: "24px",
                    fontWeight: "500",
                    color: "#666",
                }}
            >
                No Teams Available
            </Typography>
            <Typography>
                Create new team by clicking "Add Team" on top right or use
                filter
            </Typography>
            <Typography>above to view teams, if available</Typography>
        </Box>
    );
};

export default NoData;
