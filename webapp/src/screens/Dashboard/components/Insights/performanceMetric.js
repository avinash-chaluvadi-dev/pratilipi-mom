import React from "react";
import Box from "@material-ui/core/Box";
import Typography from "@material-ui/core/Typography";

import excellent from "static/images/excellent.png";
import good from "static/images/good.png";
import medium from "static/images/medium.png";
import poor from "static/images/poor.png";
import globalSyles from "styles";

const PerformanceMetric = ({ performance }) => {
    const globalClasses = globalSyles();
    const getColor = () => {
        if (performance === "Excellent") return "#1665DF";
        else if (performance === "Good") return "#34B53A";
        else if (performance === "Medium") return "#FFB200";
        else return "#FA3E3E";
    };
    const getIcon = () => {
        if (performance === "Excellent")
            return <img src={excellent} alt="Excellent" />;
        else if (performance === "Good") return <img src={good} alt="Good" />;
        else if (performance === "Medium")
            return <img src={medium} alt="Medium" />;
        else return <img src={poor} alt="Poor" />;
    };

    return (
        <Box
            style={{ color: getColor(), fontWeight: "bold" }}
            className={globalClasses.flex}
        >
            {getIcon()}{" "}
            <Typography style={{ marginLeft: "10px" }}>
                {" "}
                {performance}
            </Typography>
        </Box>
    );
};

export default PerformanceMetric;
