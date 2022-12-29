import React, { useState, useRef } from "react";
import DatePicker from "react-multi-date-picker";
import { Button } from "@material-ui/core";
import RangeDPStyles from "components/RangeDP/styles";
import ArrowForwardSharpIcon from "@mui/icons-material/ArrowForwardSharp";
import ArrowBackSharpIcon from "@mui/icons-material/ArrowBackSharp";
import CalenderIcon from "static/images/calenderIcon.svg";
import { TextField, IconButton } from "@material-ui/core";

const RangeDP = (props) => {
    let { handleDateChange, inputWidth, isRange, currentValue } = props;
    const [val, setVal] = useState(currentValue);
    const ref = useRef();
    const [shouldCloseCalendar, setShouldCloseCalendar] = useState(false);
    const weekDays = ["SU", "M", "TU", "W", "TH", "F", "SA"];
    const months = [
        "JANUARY",
        "FEBRUARY",
        "MARCH",
        "APRIL",
        "MAY",
        "JUNE",
        "JULY",
        "AUGUST",
        "SEPTEMBER",
        "OCTOBER",
        "NOVEMBER",
        "DECEMBER",
    ];
    const handleValChange = (data) => {
        setVal(data);
    };

    const BottomPlugin = ({ onClose }) => {
        const classes = RangeDPStyles();
        return (
            <div style={{ textAlign: "right" }}>
                <Button
                    variant="outlined"
                    color="primary"
                    className={classes.applyBtnCls}
                    onClick={() => {
                        setVal(undefined);
                        setShouldCloseCalendar(true);
                        setTimeout(() => {
                            ref.current.closeCalendar();
                        }, 20);
                    }}
                >
                    Cancel
                </Button>
                <Button
                    variant="contained"
                    color="primary"
                    className={classes.applyBtnCls}
                    onClick={() => {
                        handleDateChange(val);
                        setShouldCloseCalendar(true);
                        setTimeout(() => {
                            ref.current.closeCalendar();
                        }, 20);
                    }}
                >
                    Apply
                </Button>
            </div>
        );
    };

    const CustomButton = ({ direction, handleClick, disabled }) => {
        return (
            <i
                onClick={handleClick}
                style={{
                    padding: "0 10px",
                    fontWeight: "bold",
                    color: disabled ? "gray" : "#286ce2",
                    fontSize: "20px",
                }}
                className={disabled ? "cursor-default" : "cursor-pointer"}
            >
                {direction === "right" ? (
                    <ArrowForwardSharpIcon />
                ) : (
                    <ArrowBackSharpIcon />
                )}
            </i>
        );
    };

    const CustomInput = ({ openCalendar, value }) => {
        const classes = RangeDPStyles();
        return (
            <TextField
                onFocus={openCalendar}
                value={value}
                className={classes.inputBoxCls}
                id="standard-bare"
                variant="outlined"
                placeholder="Select Date"
                InputProps={{
                    endAdornment: (
                        <IconButton>
                            <img
                                src={CalenderIcon}
                                alt=""
                                width={inputWidth ? inputWidth : 16}
                                height={15}
                                onClick={openCalendar}
                            />
                        </IconButton>
                    ),
                }}
            />
        );
    };

    return (
        <>
            <DatePicker
                render={<CustomInput value={currentValue} />}
                range={isRange ? isRange : false}
                weekDays={weekDays}
                months={months}
                value={val}
                ref={ref}
                multiple={false}
                onChange={(dateValue) => handleValChange(dateValue)}
                format={"DD/MM/YYYY"}
                inputClass="red"
                numberOfMonths={2}
                renderButton={<CustomButton />}
                plugins={[<BottomPlugin position="bottom" />]}
                calendarPosition="top-right"
                arrow={false}
                onOpenPickNewDate={false}
                onOpen={() => setShouldCloseCalendar(false)}
                onClose={() => shouldCloseCalendar}
            />
        </>
    );
};

export default RangeDP;
