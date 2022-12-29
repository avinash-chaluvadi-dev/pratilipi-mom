import React from "react";
import Popover from "@mui/material/Popover";
import useGlobalStyles from "styles";

const SimplePopOver = ({ anchorEl, handleClose, content }) => {
    const globalStyles = useGlobalStyles();
    const open = Boolean(anchorEl);
    const id = open ? "popover" : null;
    return (
        <Popover
            className={globalStyles.Rounded}
            style={{ borderRadius: "16px" }}
            id={id}
            open={open}
            anchorEl={anchorEl}
            onClose={handleClose}
            anchorOrigin={{
                vertical: "bottom",
                horizontal: "left",
            }}
        >
            {content}
        </Popover>
    );
};

export default SimplePopOver;
