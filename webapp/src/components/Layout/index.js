import React from "react";
import clsx from "clsx";
import Sidebar from "components/Sidebar";
import { Box } from "@material-ui/core";
import useStyles from "./useStyles";
import globalStyles from "styles";
import { useSelector } from "react-redux";

const Layout = ({ children }) => {
    const [open, setOpen] = React.useState(false);
    const { userAuthenticated } = useSelector(
        (state) => state.userLoginReducer
    );

    const classes = useStyles();
    const globalClasses = globalStyles();
    return (
        <Box className={globalClasses.bgLight}>
            {/* {userAuthenticated ? <Sidebar open={open} setOpen={setOpen} /> : ''} */}
            <Sidebar open={open} setOpen={setOpen} />
            <Box
                className={clsx(classes.mainContent, {
                    [classes.mainContentShift]: open,
                })}
            >
                {children}
            </Box>
        </Box>
    );
};

export default Layout;
