import React from "react";
import { Typography, Box, Button } from "@material-ui/core";
import { useDispatch } from "react-redux";
import useGlobalStyles from "styles";
import { deleteTeam } from "store/action/config";

const DeleteTeam = () => {
    return (
        <Typography
            style={{ marginTop: "20px", fontFamily: "Lato !important" }}
        >
            You cannot reverse this action once you delete it. Make sure you
            want to delete it and do not want it back again
        </Typography>
    );
};

export const DeletTeamActions = ({ handleCancel, id }) => {
    const globalStyles = useGlobalStyles();
    const dispatch = useDispatch();
    return (
        <Box display="flex" justifyContent="flex-end">
            <Button
                style={{ width: "165px" }}
                variant="outline"
                color="primary"
                className={globalStyles.SecondaryButton}
                onClick={() => handleCancel(false)}
            >
                Cancel
            </Button>
            <Button
                style={{ width: "165px", marginLeft: "20px" }}
                variant="contained"
                color="primary"
                className={globalStyles.MainButton}
                onClick={() => dispatch(deleteTeam(id))}
            >
                Delete
            </Button>
        </Box>
    );
};

export const DeleteTeamTitle = () => {
    return (
        <Box
            sx={{
                width: "570px",
            }}
        >
            Are you sure you want to delete team?
        </Box>
    );
};

export default DeleteTeam;
