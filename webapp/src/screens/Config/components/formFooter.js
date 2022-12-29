import React from "react";
import { Box, Button } from "@material-ui/core";
import useGlobalStyles from "styles";

const FormFooter = ({ handleCancel }) => {
    const globalStyles = useGlobalStyles();
    return (
        <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            className={globalStyles.TransparentBorderTop}
            pt={3}
            width="100%"
        >
            {/* <Button
        variant="outline"
        color="primary"
        className={globalStyles.SecondaryButton}
      >
        Save as Draft
      </Button> */}
            <Box></Box>
            <Box display="flex" justifyContent="space-between">
                <Button
                    variant="outline"
                    color="primary"
                    className={globalStyles.SecondaryButton}
                    onClick={() => handleCancel(false)}
                >
                    Cancel
                </Button>
                <Button
                    style={{ marginLeft: "20px" }}
                    variant="contained"
                    color="primary"
                    className={globalStyles.MainButton}
                    type="submit"
                >
                    Save
                </Button>
            </Box>
        </Box>
    );
};

export default FormFooter;
