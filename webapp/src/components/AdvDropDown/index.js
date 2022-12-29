import React, { useState } from "react";
import Select, { components } from "react-select";
import useStyles from "screens/FeedBackLoop/styles";
import labelIcon from "../../static/images/label.svg";
import entityIcon from "../../static/images/entity.svg";
import { Tooltip } from "@material-ui/core";
import userAccountIcon from "static/images/account_circle_black.svg";
import sentimentIcon from "static/images/sentiment.svg";
import { Switch, FormControlLabel } from "@mui/material";

const customDropDownRight = {
    control: (provided) => ({
        ...provided,
        border: "0px",
        fontSize: "12px",
        borderRadius: "2px",
        minHeight: "1px",
        height: "27px",
        paddingTop: "0",
        paddingBottom: "0",
        boxShadow: "none !important",
        cursor: "pointer",
        "&:hover": {
            border: "0 !important",
        },
    }),
    dropdownIndicator: (provided) => ({
        ...provided,
        minHeight: "0px",
        paddingTop: "0",
        paddingBottom: "0",
        color: "#757575",
        display: "none",
    }),
    valueContainer: (provided) => ({
        ...provided,
        minHeight: "1px",
        height: "27px",
        paddingTop: "0",
        paddingBottom: "0",
        minWidth: "20px",
    }),
    placeholder: (provided) => ({
        ...provided,
        top: "30%",
    }),
    singleValue: (provided) => ({
        ...provided,
        marginLeft: "-2px",
        top: "30%",
    }),
    menu: (provided) => ({
        ...provided,
        width: "max-content!important",
        padding: "10px",
        hyphens: "auto",
        marginTop: "10px",
        wordWrap: "break-word",
        fontSize: "14px",
        borderRadius: "10px",
        "&:hover": {
            minWidth: `max-content !important`,
            boxSizing: "border-box",
        },
    }),
    menuList: (provided, state) => ({
        ...provided,
        paddingTop: 0,
        paddingBottom: 0,
        overflow: "hidden",
        textAlign: "left",
        marginTop: 0,
    }),
    indicatorSeparator: (styles) => ({ display: "none" }),
};

const AdvDropDown = (props) => {
    let { value, onChange, options, type, isSearchable } = props;
    const classes = useStyles();
    const [checked, setChecked] = useState(false);
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const handleSwitchChange = (event) => {
        setChecked(event.target.checked);
        setIsMenuOpen(!isMenuOpen);
    };

    const ValueContainer = (props) => {
        const children = props.children;
        let Title = "",
            commonIcon = "",
            titleName = "";
        if (type === "Entity") {
            Title = "Add Entity";
            titleName = "Entities";
            commonIcon = entityIcon;
        }
        if (type === "Label") {
            Title = "Add Label";
            titleName = "Labels";
            commonIcon = labelIcon;
        }
        if (type === "AssignTo") {
            Title = "Assign To";
            titleName = "Assign To";
            commonIcon = userAccountIcon;
        }
        if (type === "sentimentIcon") {
            Title = "Add Sentiment";
            titleName = "Sentiments";
            commonIcon = sentimentIcon;
        }
        //console.log(titleName);
        return (
            components.ValueContainer && (
                <components.ValueContainer {...props}>
                    {!!children && value === null && (
                        <div>
                            {type !== "Entity" ? (
                                <Tooltip title={Title} placement="top" arrow>
                                    <img
                                        src={commonIcon}
                                        height="16px"
                                        width="16px"
                                        className={classes.videoIcon}
                                        data-tip="Change Label"
                                        alt=""
                                    />
                                </Tooltip>
                            ) : (
                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={checked}
                                            onChange={handleSwitchChange}
                                            size="small"
                                        />
                                    }
                                    label="Info"
                                />
                            )}
                        </div>
                    )}
                    {children}
                </components.ValueContainer>
            )
        );
    };

    const editData = (event) => {
        props.editPopUp(event);
    };
    const { Option } = components;
    const IconOption = (props) => {
        let {
            innerProps,
            innerRef,
            isDisabled,
            isFocused,
            isSelected,
            cx,
            getStyles,
        } = props;
        // console.log(
        //   innerProps,
        //   innerRef,
        //   isDisabled,
        //   isFocused,
        //   isSelected,
        //   cx,
        //   getStyles
        // );
        return (
            <Option {...props}>
                {props.data.value !== "Add name" && isFocused && (
                    <img
                        src={props.data.icon}
                        style={{ float: "right", width: "max-content" }}
                        alt={props.data.label}
                        onClick={editData}
                    />
                )}
                {props.data.label}
            </Option>
        );
    };
    let temp = { ValueContainer };
    if (type === "AssignTo") {
        temp = { ValueContainer, ...{ Option: IconOption } };
    }
    return (
        <Select
            value={value}
            onChange={onChange}
            options={options}
            clearable={true}
            placeholder={""}
            className={value ? classes.dropDownRight : classes.dropDownRightCss}
            styles={customDropDownRight} //,...colourStyles}}
            components={temp}
            isSearchable={isSearchable ? true : false}
            menuIsOpen={isMenuOpen}
        />
    );
};
export default AdvDropDown;
