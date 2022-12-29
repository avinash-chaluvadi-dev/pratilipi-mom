import React from "react";
import useStyles from "./useStyles";
import { InputLabel, Box, FormHelperText } from "@material-ui/core";
import Input from "./input";

const FormInput = (props) => {
    const styles = useStyles();
    const { ml, ...other } = props;
    return (
        <Box ml={ml}>
            <InputLabel className={styles.InputLabel}>{props.label}</InputLabel>
            <Input {...other} />
            {props.showError && (
                <FormHelperText error={props.showError}>
                    {props.message}
                </FormHelperText>
            )}
        </Box>
    );
};

export default FormInput;
