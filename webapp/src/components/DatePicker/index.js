import "date-fns";
import React, { useState } from "react";
import DateFnsUtils from "@date-io/date-fns";
import useStyles from "components/DatePicker/styles";
import customStyles from "screens/FeedBackLoop/components/MoMView/useStyles";
import {
    MuiPickersUtilsProvider,
    KeyboardDatePicker,
} from "@material-ui/pickers";
import ArrowDropUpIcon from "@mui/icons-material/ArrowDropUp";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import { Grid } from "@material-ui/core";
import CalenderIcon from "static/images/calenderIcon.svg";
import dropDownIcon from "static/images/dropDown.svg";

export default function CustomDatePicker(props) {
    const classes = useStyles();
    const momCls = customStyles();
    let {
        value,
        handleDateChange,
        placeholder,
        className,
        customIcon,
        width,
        height,
    } = props;
    // The first commit of Material-UI
    const [selectedDate, setSelectedDate] = useState(value);

    // const handleDateChange = (date) => {
    //   setSelectedDate(date);
    // };
    return (
        <MuiPickersUtilsProvider utils={DateFnsUtils}>
            {/* <Grid container className={classes.datePickerGrid} justifyContent="space-around"> */}
            <KeyboardDatePicker
                disableToolbar
                variant="inline"
                format="yyyy-MM-dd"
                margin="0 20px"
                KeyboardButtonProps={{
                    "aria-label": "change date",
                }}
                // keyboardIcon={{color: 'red'}}
                // id="date-picker-inline"
                // inputVariant="outlined"
                value={value}
                onChange={handleDateChange}
                placeholder={placeholder ? placeholder : "Date"}
                InputAdornmentProps={{ position: "end" }}
                className={className ? className : classes.datePickerGrid}
                autoOk
                InputProps={{ color: "red" }}
                // InputProps={{
                //   endAdornment: (
                //     <InputAdornment position="end">
                //     </InputAdornment>
                //   ),
                // }}
                keyboardIcon={
                    customIcon ? (
                        <ArrowDropDownIcon />
                    ) : (
                        <img
                            src={CalenderIcon}
                            alt=""
                            width={width}
                            height={height}
                        />
                    )
                }
            />
            {/* </Grid> */}
        </MuiPickersUtilsProvider>
    );
}
