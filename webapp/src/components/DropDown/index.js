import React from "react";
import Select, { components } from "react-select";
import useStyles from "screens/FeedBackLoop/styles";
import { makeStyles } from "@material-ui/core/styles";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";

const customStyles = {
    control: (provided) => ({
        ...provided,
        border: "0.5px solid #D8D8D8;",
        borderRadius: "2px",
        minHeight: "1px",
        height: "27px",
        paddingTop: "0",
        paddingBottom: "0",
        boxShadow: "none",
        cursor: "pointer",
        "&:hover": {
            border: "1px solid #D8D8D8",
        },
    }),
    dropdownIndicator: (provided) => ({
        ...provided,
        minHeight: "1px",
        paddingTop: "0",
        paddingBottom: "0",
        color: "#757575",
    }),
    valueContainer: (provided) => ({
        ...provided,
        minHeight: "1px",
        height: "27px",
        paddingTop: "0",
        paddingBottom: "0",
    }),
    placeholder: (provided) => ({
        ...provided,
        top: "40%",
    }),
    singleValue: (provided) => ({
        ...provided,
        marginLeft: "-2px",
        top: "40%",
    }),
    menu: (provided) => ({
        ...provided,
        borderRadius: 0,
        width: "max-content!important",
    }),
    menuList: (provided, state) => ({
        ...provided,
        paddingTop: 0,
        paddingBottom: 0,
        overflow: "hidden",
    }),
};

const useStylesInternal = makeStyles((theme) => ({
    dropDown: {
        width: ({ widthValue }) => (widthValue ? widthValue : "100%"),
        color: ({ type }) => "#333333",
        fontSize: "14px",
        fontWeight: "bold",
    },
}));

const DropDown = (props) => {
    const defaultWidth = props.value
        ? props.value?.value
        : props.placeholder
        ? props.placeholder
        : "";
    let charCount = (defaultWidth + `(${0})`).length;
    let widthValue = "";
    if (props.from === "summary") {
        widthValue = `${7 * charCount + (charCount >= 8 ? 14 : 34)}px`;
    } else {
        widthValue = `${7 * charCount + (charCount >= 8 ? 10 : 35)}px`;
    }
    const classes = useStyles();
    let type = props.type;
    const classesInt = useStylesInternal({ widthValue, type });

    const customDropDownRight = {
        control: (provided, { isDisabled }) => ({
            ...provided,
            border: "0px",
            fontSize: "14px",
            fontWeight: "bold",
            borderRadius: "2px",
            minHeight: "1px",
            height: "25px",
            boxShadow: "none !important",
            background: "inherit",
            cursor: isDisabled ? "not-allowed" : "pointer",
            opacity: isDisabled ? "0.6" : "",
            "&:hover": {
                border: "0 !important",
                background: "#EAEAEA",
                paddingTop: "0",
                paddingBottom: "0",
                height: "20px",
            },
        }),
        dropdownIndicator: (provided) => ({
            ...provided,
            minHeight: "0px",
            color: "#757575",
            // display: 'none'
            margin: "-10px 0 0 -20px",
            padding: "0 0 0 0",
        }),
        valueContainer: (provided) => ({
            ...provided,
            minHeight: "1px",
            height: "27px",
            paddingTop: "0",
            paddingBottom: "0",
            minWidth: "20px",
            padding: "2px 1px 1px 1px !important",
            margin: "3px 7px",
            // display: "inline-block"
        }),
        placeholder: (provided) => ({
            ...provided,
            top: "30%",
            // overflow: 'hidden',
            // whiteSpace: 'nowrap',
            // textOverflow: 'ellipsis',
            color: colorCheck(props.type),
        }),
        singleValue: (provided) => ({
            ...provided,
            top: "30%",
            color:
                props?.value?.length > 1 ? colorCheck(props.type) : "inherit",
        }),
        menu: (provided) => ({
            ...provided,
            width: `max-content !important`,
            hyphens: "auto",
            marginTop: "-1px!important",
            borderRadius: "4px",
            fontSize: "14px",
            fontFamily: "Lato",
            fontWeight: "normal",
            padding: "5px",
            "&:hover": {
                minWidth: `max-content !important`,
                boxSizing: "border-box",
            },
            "&:focused": {
                minWidth: `max-content !important`,
                boxSizing: "border-box",
            },
        }),
        menuList: (provided, state) => ({
            ...provided,
            paddingTop: 0,
            paddingBottom: 0,
            overflow: props.isScroll ? "scroll" : "hidden",
            textAlign: "left",
            marginTop: 0,
        }),
        indicatorSeparator: (styles) => ({
            display: "none",
            margin: "0",
            padding: "0",
        }),
        option: (styles, { isFocused, isSelected }) => ({
            ...styles,
            background: isSelected ? "#056aea" : undefined,
            borderTop: isSelected ? "0.5px solid #FFFFFF" : undefined,
            zIndex: 1,
        }),
        container: (provided) => ({
            ...provided,
            width: 100,
            // flex: 1,
        }),
    };

    const DropdownIndicator = (props) => {
        return (
            <components.DropdownIndicator {...props}>
                <ArrowDropDownIcon />
            </components.DropdownIndicator>
        );
    };

    const colorCheck = (type) => {
        switch (type) {
            case "Label":
                return "#DFAA20";
            case "Entity":
                return "#372EC1";
            case "AssignTo":
                return "#49ce40";
            case "sentimentIcon":
                return "#FFC133";
            default:
                return "";
        }
    };

    return (
        <Select
            value={
                // props?.value?.length > 1
                //   ? [
                //       {
                //         label: props.placeholder + `(${props?.value.length})`,
                //         value: props.placeholder + `(${props?.value.length})`,
                //       },
                //       ...props.value,
                //     ]
                //   : props.value
                props.value
            }
            autoFocus
            onChange={props.onChange}
            options={props.options}
            placeholder={props.placeholder ? props.placeholder : "Select"}
            clearable={true}
            className={
                props.isNormal
                    ? `${classes.dropDown} ${
                          props.disabled ? classes.disabledCls : {}
                      } `
                    : classesInt.dropDown
            }
            styles={props.isNormal ? customStyles : customDropDownRight}
            isSearchable={props.isSearchable ? true : false}
            components={{ DropdownIndicator }}
            multiple={true}
            isDisabled={props.disabled ? true : false}
        />
    );
};

export default DropDown;
