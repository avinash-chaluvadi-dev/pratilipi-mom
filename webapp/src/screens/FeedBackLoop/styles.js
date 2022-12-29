import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    feedbackloop: {
        width: "100%",
        height: "539px",
        padding: "20px",
        marginBottom: "6px",
        backgroundColor: theme.palette.white.main,
    },
    btnColor: {
        borderColor: "red",
        color: "red",
        fontWeight: "bold",
        cursor: "pointer",
        textTransform: "none",
    },
    btnColorGreen: {
        cursor: "pointer",
        textTransform: "none",
        color: "#81cc2d",
        borderColor: "#81cc2d!important",
        fontWeight: "bold",
    },
    playIcon: {
        float: "left",
        padding: "6px 12px",
        color: "primary", //'#000000',
    },
    cardPadding: {
        borderRadius: "16px",
        boxShadow: "0 16px 32px 0 rgba(0, 0, 0, 0.1)",
        border: "solid 1px rgba(0, 0, 0, 0.08)",
        backgroundColor: "#fff",
        padding: "19px 22px 19px 19px",
        height: "1149px", //'70vw',
        width: "1350px", //'88vw',
        margin: "0 0 20px 0",
    },
    marginTopHeader: {
        marginTop: "5px",
        float: "left",
        color: "primary", //'#000000',
    },
    paperSecond: {
        padding: theme.spacing(2),
        margin: "0px 0px 0px 55px",
        textAlign: "center",
        color: theme.palette.text.secondary,
        boxShadow: "none",
        border: "2px solid #D8D8D8",
        maxWidth: "46%",
        height: "80vh",
        overflowY: "scroll",
    },
    reviewMomBtn: {
        marginTop: "5px",
        float: "right",
        width: "145px",
        height: "30px",
        margin: "5px",
        borderRadius: "8px",
        backgroundBlendMode: "source-in",
        backgroundImage:
            "linear-gradient(to bottom, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.1))",
        textTransform: "none",
        marginRight: "-60px",
        fontFamily: "Lato",
        fontSize: "14px",
        fontWeight: "bold",
    },
    previewMomBtn: {
        marginTop: "5px",
        float: "right",
        width: "145px",
        height: "30px",
        margin: "5px",
        borderRadius: "8px",
        fontSize: "14px",
        fontWeight: "bold",
        fontFamily: "Lato",
        textTransform: "none",
        // width: '48%',
    },
    overviewBlock: {
        background: "#F8F8F8",
        borderRadius: "8px",
        margin: "7px",
    },
    overViewTitle: {
        fontSize: "12px",
        fontWeight: "bold",
        color: "#8c8c8c",
        padding: "20px",
        paddingBottom: "5px",
        textAlign: "left",
    },
    rootGrid: {
        flexGrow: 1,
    },
    root: {
        cursor: "pointer",
        color: "#333333",
        fontFamily: "Lato",
        fontSize: "12px",
        fontWeight: "bold",
        fontStretch: "normal",
        fontStyle: "normal",
        "& .MuiInputBase-root": {
            borderColor: "#F8F8F8",
            color: "#333333",
            fontFamily: "Lato",
            fontSize: "12px",
            fontWeight: "bold",
        },
        "& .MuiInputBase-root.Mui-focused": {
            borderColor: "gray",
            background: "white",
            color: "#333333",
        },
        "& .MuiOutlinedInput-root": {
            "& fieldset": {
                borderColor: "#F8F8F8",
            },
            "&:hover fieldset": {
                borderColor: "#F8F8F8",
                cursor: "pointer",
            },
            "&.Mui-focused fieldset": {
                borderColor: "gray",
                borderWidth: "1px",
                color: "black",
            },
        },
        "& .MuiOutlinedInput-multiline": {
            padding: "20px",
            paddingTop: "5px",
        },
    },
    tabsView: {
        textAlign: "left",
        margin: "26px 5px",
    },
    dividerStyle: {
        margin: "12px 0px 11px 0px",
    },
    dropDown: {
        width: "16% !important",
        margin: "0px 11px",
    },
    disabledCls: {
        opacity: "0.4",
    },
    button: {
        width: "fit-content",
        height: "26px",
    },
    horizontalDividerDiv: {
        margin: "14px 0px 35px 0px",
    },
    horizontalDivider: {
        width: "98%",
        color: "#D8D8D8",
        borderTopStyle: "solid",
        borderTopWidth: "1px",
        marginLeft: "15px",
    },
    rootView: {
        flexGrow: 1,
    },
    paper: {
        padding: theme.spacing(2),
        margin: "0px 0px 0px 28px",
        textAlign: "center",
        color: theme.palette.text.secondary,
        boxShadow: "none",
        // border: '2px solid #D8D8D8',
        maxWidth: "46%",
        height: "80vh",
        overflowY: "scroll",
    },
    paperCommonVal: {
        padding: theme.spacing(2),
        margin: theme.spacing(1),
        textAlign: "center",
        color: theme.palette.text.secondary,
        border: "1px solid #CFCFCF",
        height: "50px",
        boxShadow:
            "0px 1px 1px -1px rgb(0 0 0 / 20%), 0px 1px 1px 0px rgb(0 0 0 / 14%), 0px 1px 3px 0px rgb(0 0 0 / 12%)",
    },
    paperCommonHead: {
        color: "#3A4159",
        padding: "16px 33px 16px 18px",
        textAlign: "left",
        border: "1px solid rgba(0, 0, 0, 0.1)",
        borderRadius: "10px",
        fontWeight: "bold",
        fontSize: "18px",
        position: "absolute",
        height: "95vh", //'83.4vh',
        width: "600px",
        background: "transparent",
    },
    paperCommonHeadSecond: {
        color: "#3A4159",
        padding: "16px 33px 16px 18px",
        textAlign: "left",
        border: "1px solid rgba(0, 0, 0, 0.1)",
        borderRadius: "10px",
        fontWeight: "bold",
        fontSize: "18px",
        position: "absolute",
        height: "95vh", //'83.4vh',
        width: "600px",
        background: "transparent",
        margin: "0px 0px 9px 4px",
    },

    paperFullWidth: {
        margin: "46px 18.5px 0px 26px", //-12
        textAlign: "center",
        color: theme.palette.text.secondary,
        boxShadow: "none",
        height: "90vh",
        overflowY: "scroll",
        zIndex: "0",
        borderTop: "blue!important",
        borderBottom: "white!important",
    },
    paperCommonCss: {
        padding: "20px",
        margin: "0px 23px 0px 5px",
        textAlign: "center",
        color: theme.palette.text.secondary,
        border: "1px solid rgba(0, 0, 0, 0.1)",
        borderRadius: "10px",
        height: "60px",
        boxShadow:
            "0px 1px 1px -1px rgb(0 0 0 / 20%), 0px 1px 1px 0px rgb(0 0 0 / 14%), 0px 1px 3px 0px rgb(0 0 0 / 12%)",
    },
    paperCommonCssSecond: {
        padding: "20px",
        margin: "0px -4px 0px 38px",
        textAlign: "center",
        color: theme.palette.text.secondary,
        border: "1px solid rgba(0, 0, 0, 0.1)",
        borderRadius: "10px",
        height: "60px",
        boxShadow:
            "0px 1px 1px -1px rgb(0 0 0 / 20%), 0px 1px 1px 0px rgb(0 0 0 / 14%), 0px 1px 3px 0px rgb(0 0 0 / 12%)",
    },
    paperHover: {
        // '&:hover': {
        cursor: "pointer",
        border: "2px solid #286ce2 !important",
        boxShadow: "0px 7px 7px rgba(0,0,0,0.3), 0px 10px 10px rgba(0,0,0,0)",
        // boxShadow: '12px 0 15px -4px rgba(31, 73, 125, 0.8), -12px 0 8px -4px rgba(31, 73, 125, 0.8)'
        // boxShadow: '-10px 0 8px -8px black, 10px 0 8px -8px black'
        // },
    },
    borderCls: {
        border: "2px solid #175AC0 !important",
    },
    actionCardCss: {
        padding: "0px 15px 12px 0px !important",
    },

    tableDivider: {
        width: "108.5%",
        color: "#D8D8D8",
        borderTopStyle: "solid",
        borderTopWidth: "1px",
        margin: "15px 0px 0px -18px",
    },
    tableDividerBottom: {
        width: "98%",
        color: "#D8D8D8",
        borderTopStyle: "solid",
        borderTopWidth: "1px",
        margin: "-12px 0px 0px 6px",
    },
    tableDividerBottomLeft: {
        width: "98%",
        color: "#D8D8D8",
        borderTopStyle: "solid",
        borderTopWidth: "1px",
        margin: "-12px 0px 0px 15px",
    },
    connector: {
        width: "18%",
        position: "relative",
        margin: "-29px 0px 0px 580px", //'-42px 0px 0px 586px',
    },
    profileBar: {
        margin: "-12px 0px 0px -11px",
        float: "left",
        padding: "3px 5px 2px 5px",
        display: "flex",
    },
    profileBarLeft: {
        padding: "2px ! important",
        display: "flex",
        position: "relative",
        margin: "0 20px 2px 0px !important",
        justifyContent: "space-around",
    },
    profileBarRight: {
        padding: "2px ! important",
        display: "flex",
        position: "relative",
        margin: "0 20px 2px -10px !important",
        justifyContent: "space-between",
    },
    userName: {
        fontWeight: "bold",
        fontSize: "14px",
        padding: "0px 15px 5px 10px",
    },
    inputField: {
        margin: "0 0 0 13px",
        padding: "0 ! important",
        cursor: "pointer",
        "& .MuiInputBase-root": {
            borderColor: "#fff",
            letterSpacing: "initial",
        },
        "& .MuiInputBase-root.Mui-focused": {
            borderColor: "#D8D8D8",
        },
        "& .MuiOutlinedInput-root": {
            padding: "2px 0px 2px 2px",
            "& fieldset": {
                borderColor: "#fff",
            },
            "&:hover fieldset": {
                borderColor: "#D8D8D8",
                margin: "0px 10px 0px 0px",
                cursor: "pointer",
            },
            "&.Mui-focused fieldset": {
                borderColor: "#D8D8D8",
                borderWidth: "0.7px",
                margin: "0px 10px 0px 0px",
            },
        },
    },
    thaiTextFieldInputProps: {
        padding: "2px",
        // cursor: 'pointer'
    },
    timer: {
        fontSize: "12px",
        fontWeight: "bold",
        color: "#CCCCCC",
        padding: "0px",
        paddingTop: "2px",
        minWidth: "100px",
    },
    timerRight: {
        fontSize: "12px",
        fontWeight: "bold",
        color: "#CCCCCC",
        padding: "0px",
        paddingTop: "2px",
        minWidth: "100px",
    },
    inputFieldRight: {
        margin: "0",
        padding: "0 ! important",
        cursor: "pointer",
        "& .MuiInputBase-root": {
            borderColor: "#fff",
            letterSpacing: "initial",
        },
        "& .MuiInputBase-root.Mui-focused": {
            borderColor: "#D8D8D8",
        },
        "& .MuiOutlinedInput-root": {
            padding: "2px 0px 2px 2px",
            "& fieldset": {
                borderColor: "#fff",
            },
            "&:hover fieldset": {
                borderColor: "#D8D8D8",
                margin: "0px 10px 0px 0px",
                cursor: "pointer",
            },
            "&.Mui-focused fieldset": {
                borderColor: "#D8D8D8",
                borderWidth: "0.7px",
                margin: "0px 10px 0px 0px",
            },
        },
    },
    videoIcon: {
        padding: "0px",
        paddingLeft: "10px",
        paddingTop: "2px",
        cursor: "pointer",
    },
    openPopup: {
        right: -20,
        position: "absolute",
    },
    confindencecls: {
        position: "absolute",
        height: "18px",
        background: "#C6DCFF",
        borderRadius: "4px",
        color: "#1665DF",
        margin: "0 0 0 413px",
    },
    dropDownRight: {
        width: "100px ",
    },
    dropDownRightCss: {
        margin: "-2px -3px",
    },
    spanMargin: {
        margin: "-2px 5px 5px 5px",
        textAlign: "center",
    },
    confindencesummary: {
        position: "absolute",
        height: "18px",
        background: "#C6DCFF",
        borderRadius: "4px",
        color: "#1665DF",
        margin: "0 0 0 343px",
    },
    imgPadding: {
        paddingBottom: "-10px!important",
    },
    datePickerGrid: {
        // width: "15%",
        margin: "1px 2px 10px 3px",
        border: "0.8px solid #EAEAEA",
        color: "#645BF5",
        "&:hover": {
            borderColor: "#EAEAEA",
            background: "#EAEAEA",
            cursor: "pointer",
        },
        "& .MuiIconButton-root": {
            padding: "0px",
            margin: "0px",
        },
        "& .MuiSvgIcon-root": {
            width: "0.7em",
            height: "0.7em",
        },
        "& .MuiInputBase-input": {
            padding: "0px",
            margin: "0px",
            minWidth: "10px",
            maxWidth: "68px",
            color: "#645BF5",
            // fontSize:'12px',
        },
        "& .MuiInputAdornment-positionStart": {
            margin: "0px",
        },
        "& .MuiInputAdornment-positionEnd": {
            margin: "0px",
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
    calenderIcon: {
        color: "#645BF5",
    },
    paperModalPopup: {
        padding: theme.spacing(2),
        margin: "0px 0px 0px 15px",
        textAlign: "center",
        color: theme.palette.text.secondary,
        // border: '1px solid #CFCFCF',
        height: "50px",
        background: "#F8F8F8",
        width: "100%",
        boxShadow: "none",
    },
    modalWidtHeight: {
        "& .MuiDialog-paperWidthMd": {
            width: "838px",
            maxWidth: "838px",
            height: "504px",
        },
        "& .MuiDialog-paperFullWidth": {
            width: "838px",
            maxWidth: "838px",
            height: "504px",
        },
    },
    modalTitle: {
        borderRadius: "16px",
        height: "50px",
        background: "#f7f7f7 padding-box",
    },
    modalTitleBar: {
        margin: "20px 0px 0px 0px",
    },
    modalContentBar: {
        margin: "0px 0px 20px 0px",
    },
    modalContentTitleBar: {
        margin: "20px 0px 20px 0px",
    },
    modalContentImageBar: {
        margin: "5px 0px 5px 0px",
    },
    textOverflow: {
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
    },
    playbtn: {
        background: "#FFFFFF",
        border: "2px solid #f0f5ff",
        boxSizing: "border-box",
        borderRadius: "8px",
        float: "left",
        height: "30px",
        margin: "8px 0 0 6px",
        color: "#1665DF",
        width: "fit-content",
        textTransform: "none",
        fontFamily: "Lato",
        fontSize: "14px",
        fontWeight: "bold",
        "&:hover": {
            backgroundColor: "rgba(240, 245, 255, 0.5)",
            color: "#286ce2",
            // boxShadow: 'none',
        },
        "&:active": {
            // boxShadow: 'none',
            backgroundColor: "#1665DF",
            color: "#ffff",
        },
    },

    filterbtn: {
        background: "#FFFFFF",
        border: "1px solid #949494",
        borderRadius: "8px",
        height: "40px",
        margin: "20px 0 30px 0px",
        color: "#286ce2",
        fontSize: "14px",
        width: "250px",
        fontWeight: "bold",
        justifyContent: "space-between",
        "&:hover": {
            backgroundColor: "#FFFFFF",
            color: "#286ce2",
            boxShadow: "none",
        },
        "&:active": {
            boxShadow: "none",
            backgroundColor: "#FFFFFF",
            color: "#286ce2",
        },
    },
    select: {
        fontSize: "14px",
        fontWeight: "bold",
        paddingRight: "5px",
        borderRadius: "20px",
        width: "260px",
    },

    btnRight: {
        margin: "0 0 0 39.5rem",
        textTransform: "none",
    },
    transformvalue: {
        textTransform: "none",
        fontSize: "14px",
    },
    confidencegreenscore: {
        borderRight: "5px solid #34B53A ! important",
    },
    confidenceyellowscore: {
        borderRight: "5px solid #FFB200 ! important",
    },
    confidenceredscore: {
        borderRight: "5px solid #FA3E3E ! important",
    },
    headecls: {
        display: "flex",
        flexDirection: "row",
        flexWrap: "wrap",
        alignContent: "stretch",
        justifyContent: "flex-start",
        alignItems: "center",
    },
    headeTextcls: {
        cursor: "pointer",
        color: "black",
        "& .MuiInputBase-root": {
            borderColor: "#F8F8F8",
            color: "black",
            fontSize: "16px",
            fontWeight: "bold",
            fontFamily: "Lato",
        },
        "& .MuiInputBase-root.Mui-focused": {
            borderColor: "gray",
            background: "white",
            color: "black",
        },
        "& .MuiOutlinedInput-root": {
            "& fieldset": {
                borderColor: "#F8F8F8",
            },
            "&:hover fieldset": {
                borderColor: "#F8F8F8",
                cursor: "pointer",
            },
            "&.Mui-focused fieldset": {
                borderColor: "gray",
                borderWidth: "1px",
                color: "black",
            },
        },
    },
    sharemom: {
        display: "flex",
        flexDirection: "row",
        justifyContent: "flex-end",
    },
    tooltip: {
        fontSize: "10px",
        fontWeight: "bold",
        paddingLeft: "14px",
        paddingRight: "14px",
        alignItems: "center",
        justifyContent: "center",
    },
}));

export default useStyles;
