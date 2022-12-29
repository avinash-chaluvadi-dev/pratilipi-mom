import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    listroot: {
        // width: '100%',
        // maxWidth: 360,
        float: "left",
        background: "#FFFFFF",
        // border: '1px solid #D8D8D8',
        boxSizing: "border-box",
        height: "617px",
        width: "210px",
        margin: "-8.5px -8px",
        marginBottom: "-7px",
        paddingTop: "0px",
    },
    gridLelt: {
        marginLeft: "13px",
    },
    mainGrid: {
        background: "#F8F8F8",
        border: "2px solid #D8D8D8",
        boxSizing: "border-box",
        // borderRadius: '4px',
        margin: "20px 0 0 14px",
    },
    listItems: {
        background: "#FFFFFF",
        border: "1px solid #D8D8D8",
        boxSizing: "border-box",
        // MuiListItem-root.Mui-selected, .MuiListItem-root.Mui-selected:hover {
    },
    listItemSelected: {
        borderRight: "0px!important",
        boxSizing: "border-box",
        background: "#F8F8F8!important",
        backgroundColor: "#F8F8F8!important",
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
        height: "50px",
        boxShadow:
            "0px 1px 1px -1px rgb(0 0 0 / 20%), 0px 1px 1px 0px rgb(0 0 0 / 14%), 0px 1px 3px 0px rgb(0 0 0 / 12%)",
    },
    paperCommonCssSecond: {
        padding: theme.spacing(2),
        // margin: '0px -4px 0px 38px',
        textAlign: "center",
        color: theme.palette.text.secondary,
        border: "1px solid #CFCFCF",
        height: "50px",
        boxShadow:
            "0px 1px 1px -1px rgb(0 0 0 / 20%), 0px 1px 1px 0px rgb(0 0 0 / 14%), 0px 1px 3px 0px rgb(0 0 0 / 12%)",
    },
    headercls: {
        paddingLeft: "21px !important",
    },
    boldStyle: {
        fontWeight: "bold",
    },
    connectorSummary: {
        width: "8%",
        position: "relative",
        margin: "-30px 0px 0px 494px",
    },
    paperHoverStyle: {
        // '&:hover': {
        cursor: "pointer",
        boxShadow: "0px 7px 7px rgba(0,0,0,0.3), 0px 10px 10px rgba(0,0,0,0)",
        // boxShadow: '12px 0 15px -4px rgba(31, 73, 125, 0.8), -12px 0 8px -4px rgba(31, 73, 125, 0.8)'
        // boxShadow: '-10px 0 8px -8px black, 10px 0 8px -8px black'
        // },
    },
    userText: {
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
                paddingLeft: "2px",
                paddingTop: "2px",
                fontWeight: "bold",
                fontSize: "12px",
                color: "#333333",
            },
        },
    },
    sortcls: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        color: "#175AC0",
        fontSize: "14px",
    },
    participants: {
        textTransform: "none",
        margin: "0 23px",
        fontSize: "15px",
        "&:hover": {
            // background: 'gray',
            // padding: '0 5px',
        },
    },
    addTeamPopUp: {
        padding: "5px",
        width: "300px",
        height: "100px",
        position: "absolute",
        top: "380px",
        left: "338px",
        zIndex: 1,
    },
}));

export default useStyles;
