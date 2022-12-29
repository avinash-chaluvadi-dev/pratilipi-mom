import { makeStyles } from "@material-ui/core/styles";

const drawerWidth = 228;
const useStyles = makeStyles((theme) => ({
    appBar: {
        width: `calc(100% - ${theme.spacing(9) + 1}px)`,
        height: "64px",
        backgroundColor: theme.palette.common.white,
        color: theme.palette.text.primary,
        transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
        }),
    },
    appBarShift: {
        marginLeft: drawerWidth,
        width: `calc(100% - ${drawerWidth}px)`,
        transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    },
    drawer: {
        width: drawerWidth,
        flexShrink: 10,
        whiteSpace: "nowrap",
    },
    drawerOpen: {
        width: drawerWidth,
        backgroundColor: theme.palette.primary.main,
        transition: theme.transitions.create("width", {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    },
    drawerClose: {
        backgroundColor: theme.palette.primary.main,
        transition: theme.transitions.create("width", {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
        }),
        overflowX: "hidden",
        width: theme.spacing(9) + 1,
        borderTopRightRadius: "16px",
    },
    sidebarMenu: {
        color: theme.palette.common.white,
        borderLeft: "5px solid #1464DF",
    },
    sidebarMenuSelected: {
        color: theme.palette.common.white,
        borderLeft: "5px solid #ffffff",
    },
    divider: {
        margin: "0px 12px",
        height: "4vh",
    },
    logoutDropdown: {
        marginLeft: "-20px",
    },
    logoutIconStyles: {
        marginRight: "12px",
    },
    logoutTextContainer: {
        color: "#000000de",
        width: "180px",
    },
    logoutTextOne: {
        fontSize: "15px",
        marginRight: "4px",
    },
    logoutText: {
        fontSize: "15px",
    },
}));

export default useStyles;
