import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
    root: {
        display: "flex",
        flexDirection: "column",
        alignItems: "left",
    },
    formControl: {
        margin: theme.spacing(1),
        minWidth: 200,
    },
    buttonIcon: {
        height: "25px",
    },
    statsCard: {
        borderRadius: "16px",
        padding: "20px 24px",
        width: "100%",
        boxShadow: "0 16px 32px 0 rgb(0 0 0 / 10%)",
        border: "solid 1px rgba(0, 0, 0, 0.08)",
        backgroundColor: "#fff",
    },
    statsCardMargin: {
        "&>:not(:first-child)": {
            marginLeft: "24px",
        },
    },
    transformvalue: {
        textTransform: "none",
        fontSize: "14px",
    },
    cards: {
        // position: absolute;
        // width: '406px',
        height: "500px",
        // left: '126px',
        // top: '448px',
        background: "#FFFFFF",
        border: "0.5px solid #D8D8D8",
        boxShadow: "0px 1px 2px rgba(0, 0, 0, 0.16)",
        borderRadius: "8px",
    },
    dividerCss: {
        // borderBottom: '1px solid rgba(0,0,0,0.1)',
        margin: "15px 0 19px 0",
    },
    dividerLastCard: {
        borderBottom: "2px solid rgba(0,0,0,0.1)",
        margin: "5px 20px 0 37px",
    },
    fontCss: {
        fontSize: "20px",
        color: "#2E2929",
        margin: "6px 0 0 9px",
        display: "flex",
        fontWeight: "bold",
    },
    dot: {
        height: " 17px",
        width: "17px",
        backgroundColor: "#bbb",
        borderRadius: "50%",
        margin: "5px 0 0 13px",
        // display: 'inline-block',
    },
    dotBar: {
        maxWidth: "11.33%",
        minWidth: "11.33%",
    },
    title: {
        margin: "0 0 0 25px",
    },
    borderCss: {
        height: "195px",
        overflowY: "scroll",
        margin: "0 0px 0 -15px",
    },
    paddingCss: {
        padding: "0 0 0 24px!important",
    },
    textLeft: {
        textAlign: "left",
        fontWeight: "bold",
    },
    content: {
        textAlign: "left",
        fontWeight: "500",
        fontSize: "14px",
    },
    viewallCss: {
        float: "right",
        cursor: "pointer",
        margin: "16px 15px 15px 15px",
        fontWeight: "bold",
        "&:hover": {
            color: "red",
            textDecoration: "underline",
        },
    },
    cursorCls: {
        cursor: "pointer",
        // padding: '2px 0 2px 11px',
        width: "94%",
        margin: "0 0 0 24px",
        justifyContent: "space-evenly",
        padding: "2px 0 5px 0",
        alignItems: "flex-end",
        "&:hover": {
            backgroundColor: "#c1e3f8",
            padding: "2px 0 5px 0",
            width: "94%",
        },
    },
    fontcss: {
        margin: "5px 0 0 3px",
    },
    paddingStyle: {
        margin: "2px 0 2px -6px",
    },
    cardFirstCss: {
        margin: "20px 10px",
    },
    marginCardFirstCss: {
        margin: "0 31px 0 0",
    },
    fontSizeCss: {
        fontSize: "19px",
    },
    circle: {
        width: "43px",
        height: "43px",
        borderRadius: "100%",
        background: "#FFFFFF",
        border: "2px solid #E8E8E8",
        boxSizing: "border-box",
        color: "#00000",
        textAlign: "center",
        fontSize: "14px",
        overflow: "hidden",
    },
    paddingLastCard: {
        margin: "0 37px 0 0px",
    },
    heightLastCard: {
        height: "361px",
        overflowY: "scroll",
        margin: "0 0px 0 -15px",
    },
    heightFirstCard: {
        height: "348px",
        overflowY: "scroll",
        margin: "0 0 0 8px",
    },
    widthFirstCard: {
        // margin: '0px 0px 0 37px',
        minWidth: "11%",
    },
    minWidthLastCard: {
        minWidth: "78%",
    },
    WidthLastCard: {
        minWidth: "10%",
    },
    subtextCss: {
        color: "rgba(0,0,0,0.5)",
    },
    dividerStyle: {
        margin: "12px 0px 11px 0px",
    },
    midDivider: {
        borderBottom: "2px solid rgba(0,0,0,0.1)",
    },
    titlebar: {
        color: "rgba(0,0,0,0.5)",
        padding: "0 0px 0px 22px",
    },
    titlebarinner: {
        color: "rgba(0,0,0,0.5)!important",
        margin: "0 -7px 0 27px!important",
        fontWeight: "bold!important",
        fontSize: "16px!important",
    },

    collapseScroll: {
        height: "507px",
        overflowY: "scroll",
        padding: "0 14px 0 17px",
    },
    collapseScrollinner: {
        padding: "0 17px 0 29px",
        // margin: '0 0 0 5px',
    },
    removeIcon: {
        color: "#95969A",
    },
    collapsecls: {
        borderBottom: "1.5px solid hsl(0deg 0% 80%) ! important",
    },
    collapseclsCss: {
        paddingLeft: "39px!important",
        borderBottom: "1.5px solid hsl(0deg 0% 80%) ! important",
    },
    collapseSno: {
        fontWeight: "bold",
        margin: "0px 28px 0 -7px",
        fontSize: "16px",
        color: "#3A4159!important",
    },
    titlebarinnerTitle: {
        color: "rgba(0,0,0,0.5)!important",
        fontWeight: "bold!important",
        fontSize: "16px!important",
        margin: "5px!important",
        padding: "0 0 0 9px",
    },
    collapseTitleCss: {
        fontWeight: "bold!important",
        fontSize: "16px!important",
        color: "#3A4159!important",
    },
    midInnerDivider: {
        borderBottom: "1px solid rgba(0,0,0,0.1)",
        margin: "0 0 0 10px",
    },
    profileBarVideoIcon: {
        maxWidth: "5%",
        minWidth: "5%",
    },
    profileBarDate: {
        maxWidth: "6%",
        minWidth: "6%",
    },
    firstBarCss: {
        padding: "0 10px",
    },
    sortcls: {
        justifyContent: "flex-end",
        display: "flex",
        alignItems: "center",
        color: "#175AC0",
        fontSize: "14px",
        width: "fit-content",
        border: "2px solid #d8d8d8",
        margin: "0 0 0 43px",
    },
    btnCls: {
        backgroundColor: "#fff!important",
        textTransform: "none",
        justifyContent: "flex-end",
    },
    btnColor: {
        backgroundColor: "rgb(61, 145, 255)",
        color: "white",
        marginLeft: "10px",
        "&:hover": {
            backgroundColor: "rgb(61, 145, 255)",
            color: "white",
        },
    },
    dashboardListCard: {
        // width: '370.3px',
        height: "250px",
        margin: "24px 24.7px 0 0px",
        padding: "24px 24px 24px 24px",
        borderRadius: "16px",
        border: "solid 1px rgba(0, 0, 0, 0.08)",
        backgroundColor: "#fff",
    },
    more: {
        margin: "0 6px 0 0",
        fontFamily: "Lato",
        fontSize: "16px",
        fontWeight: "normal",
        fontStretch: "normal",
        fontStyle: "normal",
        lineHeight: "normal",
        letterSpacing: "normal",
        textAlign: "right",
        color: "#286ce2",
        // padding: '122px 0 0 0',
    },
    mainDashboardCard: {
        borderRadius: "16px",
        boxShadow: "0 16px 32px 0 rgba(0, 0, 0, 0.1)",
        border: "solid 1px rgba(0, 0, 0, 0.08)",
    },
    titleText: {
        fontSize: "16px!important",
        paddingLeft: "10px",
    },
    titleBar: {
        display: "flex",
        justifyContent: "space-between",
        padding: "24px 24.7px 0 27.7px",
    },
    insights: {
        fontSize: "20px!important",
        fontWeight: "bold!important",
        color: "#464546!important",
    },
    insightsBody: {
        fontFamily: "Bitter!important",
        fontSize: "20px!important",
        fontWeight: "500!important",
        color: "#333!important",
    },
    popupSubmitBtn: {
        marginTop: "5px",
        float: "right",
        width: "145px",
        height: "30px",
        margin: "5px",
        borderRadius: "8px",
        backgroundBlendMode: "source-in",
        backgroundImage:
            "linear-gradient(to bottom, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.1))",
        fontSize: "14px",
        fontWeight: "bold",
    },
    checkbox: { width: "16px", height: "16px", borderRadius: "5px" },
    innerCardTitle: {
        margin: "0 44px 16px 45px",
        fontFamily: "Lato",
        fontSize: "32px",
        fontWeight: "bold",
        fontStretch: "normal",
        fontStyle: "normal",
        lineHeight: "0.75",
        letterSpacing: "normal",
        color: "#056aea",
        padding: "60px 0 0 0",
    },
    innerCardSubTitle: {
        margin: "16px 0 0",
        fontFamily: "Lato",
        fontSize: "16px",
        fontWeight: "normal",
        fontStretch: "normal",
        fontStyle: "normal",
        lineHeight: "1.38",
        letterSpacing: "normal",
        textAlign: "center",
        color: "#333",
    },
    moreHover: {
        "&:hover": {
            color: "green",
        },
    },
    Accordianroot: {
        alignItems: "center",
    },
    Accordianheading: {
        fontSize: "18px!important",
        fontWeight: "bold!important",
        fontFamily: "Lato!important",
        color: "#666!important",
    },
    MuiAccordionSummary: {
        borderBottom: "solid 1px #979797 !important",
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
            //     border-radius: 16px;
            // box-shadow: 0 16px 32px 0 rgba(0, 0, 0, 0.1);
            // border: solid 1px rgba(0, 0, 0, 0.08);
            // background-color: #fff;
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
    modalContentBar: {
        borderBottom: "1px solid #D6D6D6",
        margin: "0px 0px 20px 0px",
    },
    modalContentTitleBar: {
        borderBottom: "1px solid #D6D6D6",
        margin: "20px 0px 20px 0px",
    },
    modalHeaderTitle: {
        fontFamily: "Bitter",
        fontSize: "24px",
        fontWeight: "500",
        fontStretch: "normal",
        fontStyle: "normal",
        lineHeight: "1.25",
        letterSpacing: "normal",
        color: "#333",
    },
}));

export default useStyles;
