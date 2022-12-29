import { makeStyles } from "@material-ui/core/styles";

const globalStyles = makeStyles((theme) => ({
    bgLight: {
        // backgroundColor: theme.palette.white.secondary,
    },
    hide: {
        display: "none",
    },
    textWhite: {
        color: theme.palette.common.white,
    },
    textDark: {
        color: theme.palette.common.black,
    },
    textMuted: {
        color: theme.palette.common.muted,
    },
    flex: {
        display: "flex",
        alignItems: "center",
    },
    spaceBetween: {
        justifyContent: "space-between",
    },
    flexStart: {
        justifyContent: "flex-start",
    },
    flexCenter: {
        justifyContent: "center",
    },
    mlAuto: {
        marginLeft: "auto",
    },
    bold: {
        fontWeight: "bold",
    },
    mt: {
        marginTop: (props) => props.margin,
    },
    ml: {
        marginLeft: (props) => props.ml,
    },
    mr: {
        marginRight: (props) => props.mr,
    },
    textColor: {
        color: (props) => props.color,
    },
    TransparentBorder: {
        borderBottom: `1px solid ${theme.palette.grey.secondary}`,
    },
    TransparentBorderTop: {
        borderTop: `1px solid ${theme.palette.grey.secondary}`,
    },
    NoBorderButton: {
        textTransform: "none",
        width: "185px",
        height: "30px",
        fontSize: "16px",
        borderRadius: "8px",
        border: "none",
        "&:hover": { border: "none" },
    },
    SecondHeading: {
        fontSize: "14px",
        fontFamily: "Lato",
        fontWeight: "bold",
        color: theme.palette.dark.primary,
    },
    MainButton: {
        textTransform: "none",
        width: "180px",
        height: "35px",
        fontSize: "16px",
        fontWeight: "bold",
        borderRadius: "8px",
        backgroundImage:
            "linear-gradient(to bottom, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.1))",
    },
    SecondaryButton: {
        textTransform: "none",
        color: theme.palette.primary.main,
        width: "180px",
        height: "35px",
        fontSize: "16px",
        fontWeight: "bold",
        borderRadius: "8px",
        border: `2px solid ${theme.palette.primary.tertiary}`,
    },
    Rounded: {
        "& .MuiPaper-rounded": {
            borderRadius: "16px",
        },
    },
}));

export default globalStyles;
