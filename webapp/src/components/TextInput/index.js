import { TextField } from "@material-ui/core";
import useStyles from "screens/FeedBackLoop/styles";

const TextInput = (props) => {
    const classes = useStyles();
    // console.log('===event=====', props);

    return (
        <TextField
            autoFocus
            autoSave
            autoCorrect
            required
            type="text"
            multiline
            rows={2}
            margin="normal"
            variant="outlined"
            size={"small"}
            fullWidth
            placeholder={
                props.placeholder ? props.placeholder : "Enter Text Here"
            }
            value={
                props.value
                // "Code Walk-Through. Is it a File? The Divesh App Launches Today. You Know, I'm Happy That We Are Moving Right Now. ServiceNow Barton - Is There Any Task Addition?"
            }
            className={classes.inputField}
            InputProps={{
                classes: {
                    input: classes.thaiTextFieldInputProps,
                },
            }}
            ref={props.ref}
            onChange={props.onChange}
        />
    );
};

export default TextInput;
