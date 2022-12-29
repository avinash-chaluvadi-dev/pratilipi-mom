import "date-fns";
import React, { useState, useEffect } from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import MMdpUseStyles from "components/MMDatePicker/styles";
import CalenderIcon from "static/images/calenderIcon.svg";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import styled from "styled-components";
import { Button } from "@material-ui/core";
import moment from "moment";

const MultipleMonthDatePicker = (props) => {
    const classes = MMdpUseStyles();

    let {
        value,
        handleDateChange,
        placeholder,
        customIcon,
        width,
        height,
        type,
        disabled,
    } = props;

    const [startDate, setStartDate] = useState(null);
    const [open, setOpen] = useState(false);

    useEffect(() => {
        if (value) {
            let DateVal = moment(value, moment.defaultFormat).toDate();
            setStartDate(DateVal);
        }
    }, [value]);

    //console.log('==date===', value, startDate);
    const CustomInput = React.forwardRef((props, ref) => {
        return (
            <CustomDatePickDiv
                type={type}
                disabled={disabled}
                onClick={() => setOpen(true)}
            >
                <label ref={ref}>{props.value || props.placeholder}</label>
                {!customIcon ? (
                    <ArrowDropDownIcon />
                ) : (
                    <img
                        src={CalenderIcon}
                        alt=""
                        width={width}
                        height={height}
                        onClick={() => setOpen(true)}
                    />
                )}
            </CustomDatePickDiv>
        );
    });
    return (
        <>
            <DatePicker
                selected={startDate}
                onChange={(dateValue) => setStartDate(dateValue)}
                monthsShown={2}
                customInput={<CustomInput />}
                placeholderText={placeholder}
                className={classes.sizeCls}
                dateFormat="dd MMM yyy"
                open={open}
            >
                <div className={classes.btnGroup}>
                    <Button
                        variant="outlined"
                        color="primary"
                        className={classes.applyBtnCls}
                        onClick={() => {
                            setOpen(false);
                            setStartDate(undefined);
                        }}
                    >
                        Cancel
                    </Button>
                    <Button
                        variant="contained"
                        color="primary"
                        className={classes.applyBtnCls}
                        disabled={startDate ? false : true}
                        onClick={() => {
                            setOpen(false);
                            handleDateChange(startDate, type);
                        }}
                    >
                        Apply
                    </Button>
                </div>
            </DatePicker>
        </>
    );
};

const CustomDatePickDiv = styled.div`
    padding: 8px 10px;
    display: flex;
    align-items: center;
    border-radius: 8px;
    box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.08);
    border: solid 1px #eaeaea;
    pointer-events: ${(props) => (props.disabled === true ? "none" : "auto")};
    justify-content: ${(props) =>
        props.type === "DetailView" ? "none" : "space-between"};
    height: ${(props) => (props.type === "DetailView" ? "5px" : "21px")};
    width: ${(props) => (props.type === "DetailView" ? "105px" : "290px")};
    border-radius: ${(props) => (props.type === "DetailView" ? "none" : "8px")};
    box-shadow: ${(props) =>
        props.type === "DetailView"
            ? "none"
            : "0 2px 4px 0 rgba(0, 0, 0, 0.08)"};
    border: ${(props) =>
        props.type === "DetailView" ? "none" : "solid 1px #eaeaea"};
    font-size: 14px;
    color: ${(props) => (props.type === "DetailView" ? "#00d795" : "#286ce2")};
`;

export default MultipleMonthDatePicker;
