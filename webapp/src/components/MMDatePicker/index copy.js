import "date-fns";
import React, { useState, useEffect } from "react";
import DateFnsUtils from "@date-io/date-fns";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import MMdpUseStyles from "components/MMDatePicker/styles";
import CalenderIcon from "static/images/calenderIcon.svg";
import dropDownIcon from "static/images/dropDown.svg";
import styled from "styled-components";
import { Paper, Button } from "@material-ui/core";
import moment from "moment";

const MultipleMonthDatePicker = (props) => {
    let {
        value,
        handleDateChange,
        placeholder,
        className,
        customIcon,
        width,
        height,
        type,
    } = props;
    const date = new Date();
    // const isoDate = new Date(date.getTime() - date.getTimezoneOffset() * 60000);

    const [startDate, setStartDate] = useState(date);

    useEffect(() => {
        if (value) {
            let DateVal = moment(value, moment.defaultFormat).toDate();

            setStartDate(DateVal);
        }
    }, []);

    const classes = MMdpUseStyles();
    const CustomInput = React.forwardRef((props, ref) => {
        return (
            <CustomDatePickDiv>
                <label
                    onClick={props.onClick}
                    ref={ref}
                    onClick={props.onClick}
                >
                    {props.value || props.placeholder}
                </label>
                {!customIcon ? (
                    <img
                        src={dropDownIcon}
                        alt=""
                        width={width}
                        height={height}
                        onClick={props.onClick}
                    />
                ) : (
                    <img
                        src={CalenderIcon}
                        alt=""
                        width={width}
                        height={height}
                        onClick={props.onClick}
                    />
                )}
            </CustomDatePickDiv>
        );
    });
    return (
        <>
            <DatePicker
                selected={startDate}
                onChange={(dateValue) => handleDateChange(dateValue, type)}
                monthsShown={2}
                customInput={<CustomInput />}
                placeholderText={placeholder}
                className={classes.sizeCls}
            >
                <div className={classes.btnGroup}>
                    <Button
                        variant="outlined"
                        color="primary"
                        className={classes.applyBtnCls}
                    >
                        Cancel
                    </Button>
                    <Button
                        variant="contained"
                        color="primary"
                        className={classes.applyBtnCls}
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
    background-color: #ffffff;
    justify-content: space-between;
`;

export default MultipleMonthDatePicker;
