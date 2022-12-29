import React from "react";
import UploadPage from "./components/upload/upload";
import UploadedFiles from "./components/uploadedFiles/uploadedFiles";
import { Box, Grid } from "@material-ui/core";

const Upload = () => {
    return (
        <Grid xs={12} item={true}>
            <UploadPage />
            <Box mt={5}>
                <UploadedFiles />
            </Box>
        </Grid>
    );
};

export default Upload;
