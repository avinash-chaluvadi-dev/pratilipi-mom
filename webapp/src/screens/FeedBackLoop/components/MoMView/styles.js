import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    paper: {
        margin: "0px 0px 20px 0px",
        textAlign: "center",
        color: theme.palette.text.secondary,
        boxShadow: "none",
        border: "none",
        padding: "0 0 10px 0",
        height: "111vh",
    },
    transformvalue: {
        textTransform: "none!important",
        fontSize: "14px!important",
        fontWeight: "bold!important",
        padding: "0 20px!important",
        minHeight: "30px!important",
        maxHeight: "30px!important",
        Height: "30px!important",
    },
    paperbottom: {
        padding: theme.spacing(2),
        textAlign: "center",
        color: theme.palette.text.secondary,
        // border: '1px solid #CFCFCF',
        height: "30px",
        border: 0,
        // borderRadius: 0,
        background: "#FFFFFF",
        boxShadow: "none",
        // borderRadius: '4px 4px 0px 0px',
        // boxShadow: '0px 1px 1px -1px rgb(0 0 0 / 20%), 0px 1px 1px 0px rgb(0 0 0 / 14%), 0px 1px 3px 0px rgb(0 0 0 / 12%)',
    },
    title: {
        fontWeight: "bold",
        fontSize: "18px",
        color: "#152134",
        float: "left",
    },
    buttonCls: {
        color: "#1665DF",
        float: "right",
        textTransform: "none",
        // border: 1px solid #1665DF;
        borderRadius: "4px",
        borderColor: "#1665DF",
        "&:hover": {
            backgroundColor: "#1665DF",
            color: "#ffff",
            // boxShadow: 'none',
        },
        "&:active": {
            // boxShadow: 'none',
            backgroundColor: "#1665DF",
            color: "#ffff",
        },
    },
    labelCls: {
        float: "left",
        display: "inherit",
        margin: "0 0 5px 0",
        fontWeight: "bold",
        fontSize: "16px",
        color: "#717171",
    },
    projectblock: {
        margin: "26px 20px 0px 20px",
    },
    secondrowM: {
        margin: "5px 20px 0px 20px",
    },
    leftsidew: {
        minWidth: "48.0%",
    },
    popuptextboxleft: {
        minWidth: "66%",
    },
    secondrow: {
        minWidth: "24%",
        maxWidth: "24%",
    },
    attendeespaper: {
        // padding: theme.spacing(2),
        margin: "0px 0px 0px 0px",
        textAlign: "center",
        color: theme.palette.text.secondary,
        boxShadow: "none",
        border: "2px solid #D8D8D8",
        padding: "10px",
        // height: '34px',
        display: "flex",
    },
    attendeespaperW: {
        maxWidth: "96%",
    },
    tagcls: {
        width: "fit-content",
        background: "#F2F2F2",
        borderRadius: "23px",
        padding: "5px 8px",
        margin: "0 4px",
        cursor: "pointer",
        "&:hover": {
            background: "green",
            color: "#fff",
        },
    },
    collapsecls: {
        borderBottom: "3px solid #717171 ! important",
    },
    collapse_sub_title: {
        color: "#1665DF",
        margin: "0 59px",
        fontSize: "14px",
    },
    collapse_name: {
        "& .css-10hburv-MuiTypography-root": {
            color: "#152134!important",
            fontWeight: "bold!important",
            fontSize: "18px!important",
        },
    },
    removeIcon: {
        color: "#FF0000",
    },
    collapseScroll: {
        margin: "5px 20px 0px 20px",
        overflowY: "scroll",
        height: "440px",
        padding: "10px 0",
        width: "98.56%",
    },
    menuWidth: {
        "& .MuiMenu-list": {
            width: "160px",
            padding: "10px",
        },
        "& .MuiCheckbox-root": {
            padding: "2px 0px",
        },
        "& .MuiMenuItem-root": {
            padding: "0 0 0 4px! important",
        },
        "& .MuiListItemGutters": {
            paddingLeft: "5px",
        },
    },
    listScroll: {
        overflowY: "scroll",
        height: "200px",
    },
    listScrollTeams: {
        overflowY: "scroll",
        height: "142px",
    },
    btnMargin: {
        margin: "5px",
    },
    btngroup: {
        padding: "0!important",
    },
    datepicker: {
        border: "1px solid hsl(0, 0%, 70%)",
        borderBottom: "1px solid hsl(0, 0%, 70%)",
        borderRadius: "5px",
        "& .MuiInputBase-input": {
            fontSize: "16px",
            textAlign: "center",
        },
        "& .MuiInput-underline:before": {
            content: "none!important",
            borderBottom: "0px",
        },
        "& .MuiInput-underline:after": {
            content: "none!important",
            borderBottom: "0px",
        },
    },
    groupbtn: {
        padding: "0!important",
        margin: "10px",
    },
    assignto: {
        "& .MuiSelect-select": {
            padding: "7px 13px",
            textAlign: "left",
            "&:hover": {
                backgroundColor: "inherit!important",
                borderColor: "hsl(0, 0%, 80%)!important",
            },
            "&:focus": {
                backgroundColor: "inherit!important",
                borderColor: "hsl(0, 0%, 80%)!important",
            },
        },
        "& .MuiOutlinedInput-notchedOutline": {
            borderColor: "hsl(0, 0%, 80%)!important",
            background: "inherit!important",
        },
        "& .Mui-focused": {
            borderColor: "hsl(0, 0%, 80%)!important",
        },
    },
    addTeam: {
        color: "blue",
        fontSize: "14px",
        background: "#F2F2F2",
    },
    placeHolder: {
        color: "#717171",
        fontSize: "16px",
    },
    addTeamPopUp: {
        padding: "5px",
        width: "300px",
        height: "100px",
        position: "absolute",
        marginTop: "67px",
        zIndex: 1,
    },
    editIcon: {
        color: "#9d9d9d",
        height: "17px!important",
        width: "17px!important",
        // margin: '0 0 2px 173px',
    },
    editBlock: {
        marginRight: "0!important",
        padding: "0!important",
        borderRadius: "0!important",
        "&:hover,&:focus,&.Mui-focusVisible": {
            backgroundColor: "inherit",
            marginRight: 0,
            padding: 0,
            borderRadius: 0,
        },
    },
    userIcon: {
        color: "#9d9d9d",
        height: "17px!important",
        width: "17px!important",
        margin: "0 9px 2px 5px",
    },
    popupbtn: {
        textTransform: "none",
        maxWidth: "200px",
        maxHeight: "40px",
        minWidth: "160px",
        minHeight: "40px",
        fontSize: "16px",
        borderRadius: "8px",
    },
    modalpadding: {
        padding: "0px !important",
        "& .MuiDialogContent-root": {
            padding: "0px !important",
        },
    },
    modalContentBar: {
        borderBottom: "1px solid #D6D6D6",
        margin: "0px 0px 20px 0px",
        padding: "14px !important",
    },
    modalTitleBar: {
        borderBottom: "1px solid #D6D6D6",
        margin: "20px 0px 0px 0px",
    },
    mainpopupcontainer: {
        alignContent: "stretch",
        justifyContent: "flex-start",
        alignItems: "baseline",
        flexDirection: "row",
        padding: "10px 0 5px 0",
    },
    popupdivider: {
        display: "block!important",
        margin: "0 23px",
    },
    popuplastitem: {
        textAlign: "end",
        marginLeft: "287px",
    },
    fontweight: {
        fontWeight: "bold",
        fontSize: "16px",
    },
    contentbody: {
        background: "#FFFFFF",
        border: "0.2px solid #CFCFCF",
        boxSizing: "border-box",
        boxShadow: "0px 1px 2px rgba(0, 0, 0, 0.16)",
        borderRadius: "4px",
        margin: "5px 20px 5px 7px",
        width: "96%",
    },
    textbox: {
        height: "50px",
        padding: "2px",
        border: "none",
        "& .MuiInputBase-root": {
            borderColor: "#fff",
            letterSpacing: "initial",
        },
        "& .MuiInputBase-root.Mui-focused": {
            borderColor: "#D8D8D8",
            padding: "2px",
            height: "50px",
            border: "none",
        },
        "& .MuiOutlinedInput-root": {
            padding: "2px 0px 2px 2px",
            "& fieldset": {
                borderColor: "#fff",
                border: "none",
            },
            "&:hover fieldset": {
                borderColor: "#D8D8D8",
                margin: "0px 0px 0px 0px",
                cursor: "pointer",
                height: "50px",
                padding: "2px",
                border: "none",
            },
            "&.Mui-focused fieldset": {
                borderColor: "#D8D8D8",
                borderWidth: "0.7px",
                margin: "0px 0px 0px 0px",
                height: "50px",
                padding: "2px",
                border: "none",
            },
        },
    },
    dividercls: {
        "& .MuiDivider-root": {
            height: "1.9px",
        },
    },
    checkboxcls: {
        margin: "0 13px 0 20px",
        ".MuiCheckbox-colorPrimary.Mui-checked": {
            color: "#e82997 !important",
        },
    },
    checkboxcolor: {
        ".MuiCheckbox-colorPrimary.Mui-checked": {
            color: "#e82997 !important",
        },
    },
    fullwidthcls: {
        "& .MuiFormControl-fullWidth": {
            width: "102.8%",
        },
    },
    popupaddbtn: {
        display: "flex",
        margin: "15px 83px 15px 83px",
    },
    scrollHeight: {
        // height: '400px',
    },
    margintitle: {
        margin: "0 18px 25px 18px",
    },
    summarydetails: {
        height: "65vh",
        overflowY: "scroll",
        marginTop: "18px",
    },
    paperCommonCss: {
        padding: theme.spacing(2),
        // margin: '0px 23px 0px 5px',
        textAlign: "center",
        color: theme.palette.text.secondary,
        border: "1px solid #CFCFCF",
        height: "43px",
        boxShadow: "0px 12px 40px rgba(37, 40, 43, 0.16)",
    },
    paperCommonCssSecond: {
        padding: theme.spacing(2),
        // margin: '0px -4px 0px 38px',
        textAlign: "center",
        color: theme.palette.text.secondary,
        border: "1px solid #CFCFCF",
        height: "43px",
        boxShadow: "0px 12px 40px rgba(37, 40, 43, 0.16)",
    },
    connectorSummary: {
        width: "11%",
        position: "relative",
        margin: "-31px 0 0 382px",
    },
    actionCardCss: {
        padding: "5px 12px !important",
        minWidth: "45%",
    },
    userText: {
        width: "150px",
        height: "25px",
        margin: "0px",
        padding: "0 ! important",
        cursor: "pointer",
        "& .MuiInputBase-root": {
            borderColor: "#fff",
            letterSpacing: "initial",
            height: "25px",
        },
        "& .MuiInputBase-root.Mui-focused": {
            borderColor: "#D8D8D8",
            height: "25px",
        },
        "& .MuiOutlinedInput-root": {
            padding: "2px 0px 2px 2px",
            height: "25px",

            "& fieldset": {
                borderColor: "#fff",
                height: "25px",
            },
            "&:hover fieldset": {
                borderColor: "#D8D8D8",
                margin: "0px 10px 0px 0px",
                cursor: "pointer",
                height: "25px",
            },
            "&.Mui-focused fieldset": {
                borderColor: "#D8D8D8",
                borderWidth: "0.7px",
                margin: "0px 10px 0px 0px",
                height: "25px",
            },
            "& .MuiOutlinedInput-inputMarginDense": {
                padding: "3px",
            },
        },
    },
    projectTextcls: {
        cursor: "pointer",
        borderColor: "#d8d8d8",
        "& .MuiInputBase-root": {
            borderColor: "#d8d8d8",
        },
        "& .MuiInputBase-root.Mui-focused": {
            borderColor: "#d8d8d8",
        },
        "& .MuiOutlinedInput-root": {
            "& fieldset": {
                borderColor: "#d8d8d8",
            },
            "&:hover fieldset": {
                borderColor: "#d8d8d8",
                cursor: "pointer",
            },
            "&.Mui-focused fieldset": {
                borderColor: "#d8d8d8",
            },
        },
    },
    modaltop: {},
    buttonClsTeam: {
        padding: "7px",
        borderRadius: 0,
    },
    teambtnwidth: {
        margin: "0 -35px 0 99px",
        maxWidth: "12%!important",
        minWidth: "12%!important",
    },
    //.MuiPaper-elevation2
}));

export default useStyles;
