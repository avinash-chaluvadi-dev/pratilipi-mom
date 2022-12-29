import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    assignto: {
        "&.Mui-disabled": {
            pointerEvents: "none",
            cursor: "not-allowed",
            opacity: "0.6",
        },
        "& .MuiSelect-icon ": {
            top: "0px",
            color: "#49ce40",
            fontSize: "25px",
        },
        "& .MuiSelect-select": {
            padding: "2px 5px",
            textAlign: "left",
            fontSize: "14px",
            display: "flex",
            alignItems: "center",
            width: "75px",
            fontWeight: "bold",
            color: "#49ce40",
            "&:hover": {
                borderColor: "#EAEAEA",
                background: "#EAEAEA",
                cursor: "pointer",
            },
            "&:focus": {
                backgroundColor: "inherit!important",
                border: "none",
            },
        },
        "& .MuiOutlinedInput-notchedOutline": {
            background: "inherit!important",
            border: "none",
        },
        "& .Mui-focused": {
            border: "none",
        },
    },
    addTeam: {
        fontSize: "14px",
        fontFamily: "lato",
        fontWeight: "bold",
        width: "180px",
        color: "blue",
        background: "#F2F2F2",
    },
    menu: {
        fontSize: "14px",
        fontFamily: "lato",
        fontWeight: "bold",
        width: "180px",
        color: "#333333",
        "&:hover": {
            color: "#ffffff",
            backgroundColor: "#286ce2",
        },
    },
    placeHolder: {
        fontSize: "14px",
        fontWeight: "bold",
        color: "#49ce40",
        padding: "0 18px 0 0",
        alignItems: "center",
    },
    addTeamPopUp: {
        padding: "5px",
        width: "300px",
        height: "100px",
        position: "absolute",
        marginTop: "15px",
        zIndex: 1,
        border: "1px solid #d3cfcf",
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
            backgroundColor: "yellow",
            marginRight: 0,
            padding: 0,
            borderRadius: 0,
        },
    },
    userIcon: {
        color: "#9d9d9d",
        height: "14px!important",
        width: "14px!important",
        // margin: '0 9px 2px 5px',
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
    transcriptText: {
        margin: "0px",
        padding: "0px 0px 0px 0px! important",
        cursor: "pointer",
        height: "43px",
        overflowY: "scroll",
        width: "564px",
        textAlign: "left",
        fontFamily: "sans-serif",
        fontSize: "13.5px",
        color: "#383434",
        lineHeight: "18px",
        wordSpacing: "1px",
        borderColor: "#d3d3d3!important",
        "&:hover,&:focus,&.Mui-focusVisible,&:active": {
            borderColor: "#d3d3d3!important",
        },
        "&:active": {
            borderColor: "#d3d3d3!important",
        },
        "&[contenteditable]:focus": {
            outline: "1px solid #CFCFCF",
        },
    },

    userText: {
        margin: "0px",
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
                // height: '25px',
            },
            "&.Mui-focused fieldset": {
                borderColor: "#D8D8D8",
                borderWidth: "0.7px",
                margin: "0px 10px 0px 0px",
                // height: '25px',
            },
            "& .MuiOutlinedInput-inputMarginDense": {
                overflowY: "auto",
                width: "100%",
                textAlign: "left",
                color: "#949494",
                fontSize: "12px",
                fontWeight: "bold",
                paddingLeft: "10px",
            },
        },
    },

    userText1: {
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
            "& .MuiOutlinedInput-inputMarginDense": {
                padding: "0px",
                paddingLeft: "2px",
                paddingTop: "2px",
                fontWeight: "bold",
                fontSize: "12px",
                color: "#333333",
            },
        },
    },

    midblock: {
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-start",
        flexDirection: "row",
    },
    midblockright: {
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-end",
        flexDirection: "row",
        verticalAlign: "top",
    },
    selectioncls: {
        fontSize: "14px",
        letterSpacing: "0.25px",
        color: "#175AC0",
    },
    selectionblock: {
        background: "#E8F1FF",
        borderRadius: "4px",
        fontSize: "14px",
        lineHeight: "17px",
        letterSpacing: "0.25px",
        color: "#175AC0",
        padding: "10px!important",
        margin: "0 14px",
        display: "flex",
        alignItems: "center",
    },
    pontercls: {
        cursor: "pointer",
    },
    contextmenucls: {
        height: "250px",
        overflowY: "scroll",
        width: "13%!important",
        padding: "10px!important",
        boxShadow: "0px 12px 40px rgba(37, 40, 43, 0.16)",
        background: "#FFFFFF",
    },
    contextmenulist: {
        borderBottom: "1px solid hsl(0, 0%, 80%)!important",
    },
    borderdetail: {
        height: "42px",
        border: "none",
        overflowY: "scroll",
        cursor: "context-menu",
        width: "564px",
        padding: "0 6px",
        textAlign: "left",
        marginLeft: "10px",
        "&:hover": {
            border: "1px solid hsl(0, 0%, 80%)!important",
        },
        "&:focus": {
            border: "1px solid hsl(0, 0%, 80%)!important",
        },
    },
    button: {
        width: "fit-content",
        height: "26px",
        "&:hover": {
            background: "blue",
            backgroundColor: "blue",
            color: "white",
        },
    },
    clearfilterbutton: {
        width: "fit-content",
        height: "26px",
        "&:hover": {
            background: "red",
            backgroundColor: "red",
            color: "white",
        },
    },
    menuWidth: {
        "& .MuiPaper-root": {
            overflow: "visible",
            border: "1px solid rgba(0, 0, 0, 0.08)",
            borderRadius: "16px",
        },
        "& .MuiMenu-list": {
            height: "120px",
            width: "600px",
            padding: "0px 0px 0px 0px",
            margin: "2px",
        },
    },
    disabled: {
        pointerEvents: "none",
        cursor: "not-allowed",
    },
}));

export default useStyles;
