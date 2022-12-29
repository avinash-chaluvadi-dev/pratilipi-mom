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
        minHeight: "37px!important",
        maxHeight: "37px!important",
        Height: "37px!important",
    },
    paperbottom: {
        padding: theme.spacing(2),
        textAlign: "center",
        color: theme.palette.text.secondary,
        // border: '1px solid #CFCFCF',
        height: "30px",
        borderBottom: "solid 0.5px #949494",
        // borderRadius: 0,
        background: "#FFFFFF",
        boxShadow: "none",
        // borderRadius: '4px 4px 0px 0px',
        // boxShadow: '0px 1px 1px -1px rgb(0 0 0 / 20%), 0px 1px 1px 0px rgb(0 0 0 / 14%), 0px 1px 3px 0px rgb(0 0 0 / 12%)',
    },
    title: {
        fontWeight: "bold",
        fontSize: "20px",
        color: "#333333",
        float: "left",
        fontFamily: "Bitter",
    },
    buttonCls: {
        color: "#1665DF",
        float: "right",
        textTransform: "none",
        // border: 1px solid #1665DF;'
        borderRadius: "8px",
        fontSize: "14px",
        fontWeight: "bold",
        fontFamily: "Lato",
        border: "solid 2px #f0f5ff",
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
        fontSize: "14px",
        color: "#333333",
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
        padding: "10px",
        // height: '34px',
        display: "flex",
        borderRadius: "8px",
        boxShadow: "0 2px 4px 0 rgba(0, 0, 0, 0.08)",
        border: "solid 1px #eaeaea",
        backgroundColor: "#fff",
    },
    attendeespaperW: {
        maxWidth: "96%",
    },
    tagcls: {
        width: "fit-content",
        minWidth: "50px",
        padding: "2px 8px",
        margin: "0 4px",
        background: "#ffffff",
        borderRadius: "23px",
        height: "16px",
        fontSize: "12px",
        fontfamily: "Lato",
        fontWeight: "bold",
        color: "#056aea",
        border: "solid 1px #f0f5ff",
        "&:hover": {
            backgroundColor: "#056aea!important",
            borderColor: "#056aea!important",
            color: "#ffffff!important",
        },
    },
    collapsecls: {
        height: "60px",
        borderBottom: "solid 1px #949494 ! important",
    },
    collapse_sub_title: {
        margin: "0 59px",
        fontSize: "14px",
        color: "#286ce2",
        fontWeight: "bold",
        fontFamily: "Lato",
    },
    collapse_sub_title_disabled: {
        pointerEvents: "none",
        cursor: "not-allowed",
        margin: "0 59px",
        fontSize: "14px",
        opacity: "30%",
        filter: "grayscale(1)",
    },
    rectangle: {
        width: "8px",
        height: "8px",
        margin: "3px 8px 2px 0",
        backgroundColor: "#0061f7",
    },
    collapse_name: {
        "& .css-10hburv-MuiTypography-root": {
            color: "#333333!important",
            fontWeight: "medium!important",
            fontSize: "18px!important",
            fontFamily: "Bitter!important",
            lineHeight: "30px",
            marginLeft: "10px",
        },
    },
    sortOpenIcon: {
        padding: "6px 5px 0px 0",
        margin: "5px",
        position: "absolute",
        right: 0,
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
            padding: "20px",
            width: "220px",
        },
        "& .MuiCheckbox-root": {
            padding: "2px 0px",
        },
        "& .MuiMenuItem-root": {
            padding: "0 0 0 4px! important",
            fontSize: "14px",
            fontFamily: "Lato! important",
            lineHeight: "30px",
        },
        "& .MuiListItemGutters": {
            paddingLeft: "5px",
        },
    },
    listScroll: {},
    listScrollTeams: {
        overflowY: "scroll",
        height: "142px",
    },
    btnMargin: {
        textTransform: "none",
        width: "145px",
        height: "30px",
        margin: "5px",
        padding: "7px 0",
        borderRadius: "8px",
        fontSize: "14px",
        fontWeight: "bold",
    },
    btngroup: {
        display: "flex",
        paddingTop: "20px",
        justifyContent: "flex-end",
    },
    datepicker: {
        border: "1px solid hsl(0, 0%, 70%)",
        borderBottom: "1px solid hsl(0, 0%, 70%)",
        borderRadius: "8px",
        boxShadow: "0 2px 4px 0 rgba(0, 0, 0, 0.08)",
        color: "#286ce2",
        "& .MuiInputBase-input": {
            fontSize: "16px",
            fontWeight: "bold",
            fontStretch: "normal",
            fontStyle: "normal",
            lineHeight: "1.57",
            letterSpacing: "normal",
            textAlign: "left",
            paddingLeft: "15px",
            color: "#286ce2",
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
    rootMenuItem: {
        fontFamily: "Lato",
        fontSize: "14px",
        fontWeight: "600",
        color: "#333333",
        "&$selected": {
            backgroundColor: "#286ce2!important",
        },
        "&:hover": {
            backgroundColor: "#286ce2!important",
            borderColor: "#286ce2!important",
            color: "#ffffff!important",
        },
    },
    assignto: {
        "& .MuiSelect-select": {
            padding: "7px 13px",
            fontFamily: "Lato",
            fontSize: "14px",
            fontWeight: "600",
            textAlign: "left",
            borderColor: "#eaeaea",
            borderRadius: "8px",
            boxShadow: "0 2px 4px 0 rgba(0, 0, 0, 0.08)",
            color: "#286ce2",
            display: "flex",
            alignItems: "center",

            "&:hover": {
                backgroundColor: "inherit!important",
                borderColor: "#eaeaea",
                borderRadius: "8px",
                boxShadow: "0 2px 4px 0 rgba(0, 0, 0, 0.08)",
            },
            "&:focus": {
                backgroundColor: "inherit!important",
                borderColor: "#eaeaea",
                borderRadius: "8px",
                boxShadow: "0 2px 4px 0 rgba(0, 0, 0, 0.08)",
            },
        },
        "& .MuiOutlinedInput-notchedOutline": {
            borderColor: "#eaeaea",
            borderRadius: "8px",
            boxShadow: "0 2px 4px 0 rgba(0, 0, 0, 0.08)",
            background: "inherit!important",
        },
        "& .Mui-focused": {
            borderColor: "#eaeaea",
            borderRadius: "8px",
            boxShadow: "0 2px 4px 0 rgba(0, 0, 0, 0.08)",
        },
    },
    addTeam: {
        color: "blue",
        fontSize: "14px",
        background: "#F2F2F2",
    },
    placeHolder: {
        color: "#286ce2",
        fontSize: "14px",
        fontWeight: "bold",
        fontFamily: "Lato",
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
        backgroundColor: "#f7f7f7",
        height: "50px",
        borderRadius: "inherit",
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
        borderColor: "#eaeaea",
        borderRadius: "8px",
        boxShadow: "0 2px 4px 0 rgba(0, 0, 0, 0.08)",
        "& .MuiInputBase-root": {
            borderColor: "#eaeaea",
            fontFamily: "Lato",
            fontSize: "14px",
            fontWeight: "bold",
            color: "#286ce2",
            borderRadius: "8px",
        },
        "& .MuiInputBase-root.Mui-focused": {
            borderColor: "#eaeaea",
        },
        "& .MuiOutlinedInput-root": {
            "& fieldset": {
                borderColor: "#eaeaea",
            },
            "&:hover fieldset": {
                borderColor: "#eaeaea",
                cursor: "pointer",
            },
            "&.Mui-focused fieldset": {
                borderColor: "#eaeaea",
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
    removeSection: {
        pointerEvents: "none",
        cursor: "not-allowed",
        opacity: "30%",
        filter: "grayscale(1)",
    },
    modalOwnerName: {
        marginLeft: "5px",
        marginBottom: "5px",
        fontSize: "14px",
        fontWeight: "bold",
        fontFamily: "Lato",
    },
    //.MuiPaper-elevation2
}));

export default useStyles;
