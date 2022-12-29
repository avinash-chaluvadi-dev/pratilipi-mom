import React from "react";
import { Box } from "@material-ui/core";
import CloneIcon from "static/Icons/clone.svg";
import RemoveIcon from "static/Icons/remove.svg";
import useStyles from "../useStyles";

const MoreOptions = ({ handleClone, handleDelete }) => {
    const styles = useStyles();
    return (
        <Box
            className={styles.MoreOptions}
            display="flex"
            justifyContent="center"
            alignItems="flex-start"
            flexDirection="column"
            boxSizing="border-box"
            pl={5}
        >
            <Box
                display="flex"
                justifyContent="center"
                alignItems="center"
                style={{ cursor: "pointer" }}
                onClick={handleClone}
            >
                <Box
                    component="img"
                    src={CloneIcon}
                    style={{ color: "#949494" }}
                />
                <Box ml={2.5} mb={0.4}>
                    Clone Team
                </Box>
            </Box>
            <Box
                display="flex"
                justifyContent="center"
                alignItems="center"
                mt={2}
                style={{ cursor: "pointer" }}
                onClick={handleDelete}
                mr={4}
            >
                <Box
                    component="img"
                    src={RemoveIcon}
                    style={{ color: "#949494" }}
                />
                <Box ml={2} mb={0.4}>
                    Delete Team
                </Box>
            </Box>
        </Box>
    );
};

export default MoreOptions;
