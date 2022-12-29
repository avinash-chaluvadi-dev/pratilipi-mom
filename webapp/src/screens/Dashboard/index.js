import React from "react";
import Stats from "./components/Stats";
import Insights from "./components/Insights";
import { Box } from "@material-ui/core";

const Dashboard = () => {
    return (
        <Box display="flex" flexDirection="column" width="100%">
            <Stats />
            <Insights />
        </Box>
    );
};

export default Dashboard;
