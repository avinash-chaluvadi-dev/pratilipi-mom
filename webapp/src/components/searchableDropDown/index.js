import React from "react";
import Select, { components } from "react-select";
import useStyles from "screens/FeedBackLoop/styles";
import labelIcon from "../../static/images/label.svg";
import entityIcon from "../../static/images/entity.svg";
import { Tooltip } from "@material-ui/core";
import userAccountIcon from "static/images/account_circle_black.svg";

let val;
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
        //   boxShadow: 'none !important',
        cursor: "pointer",
        "&:hover": {
            // border: '0 !important',
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

        //   width: '90px',
        //   overflow: 'hidden',
        //   whiteSpace: 'nowrap',
        //   textOverflow: 'ellipsis'
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
        hyphens: "auto",
        marginTop: 0,
        wordWrap: "break-word",
        fontSize: "12px",
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
    const ValueContainer = (props) => {
        //console.log('====value===1====', type);
        const children = props.children;
        let Title = "",
            commonIcon = "";

        if (type === "Entity") {
            Title = "Add Entity";
            commonIcon = entityIcon;
        }
        if (type === "Label") {
            Title = "Add Label";
            commonIcon = labelIcon;
        }
        if (type === "AssignTo") {
            Title = "Assign To";
            commonIcon = userAccountIcon;
        }
        return (
            components.ValueContainer && (
                <components.ValueContainer {...props}>
                    {!!children && val === null && (
                        <div>
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
                        </div>
                    )}
                    {children}
                </components.ValueContainer>
            )
        );
    };

    return (
        <Select
            value={value}
            onChange={onChange}
            options={options}
            clearable={true}
            placeholder={""}
            className={value ? classes.dropDownRight : classes.dropDownRightCss}
            styles={customDropDownRight}
            components={{ ValueContainer }}
            isSearchable={isSearchable ? true : false}
        />
    );
};
export default AdvDropDown;
