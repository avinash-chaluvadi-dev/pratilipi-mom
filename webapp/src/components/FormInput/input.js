import React from "react";
import InputBase from "@mui/material/InputBase";
import useStyles from "./useStyles";
import { Box, Typography } from "@material-ui/core";

const FormInput = (props) => {
    const styles = useStyles();
    const { mr, ml, helperText, ...others } = props;
    return (
        <Box ml={ml} display="flex" flexDirection="column">
            <InputBase
                className={`${styles.Input} ${
                    helperText && styles.ErrorBorder
                } `}
                autoComplete="off"
                {...others}
            />
            {helperText && (
                <Typography
                    className={styles.HelperText}
                    id="passwordHelpBlock"
                >
                    {helperText}
                </Typography>
            )}
        </Box>
    );
};

export default FormInput;
