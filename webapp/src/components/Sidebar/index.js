import React, { useEffect, useState } from "react";
import { NavLink, useHistory } from "react-router-dom";
import { BroadcastChannel } from "broadcast-channel";

import clsx from "clsx";
import Drawer from "@material-ui/core/Drawer";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Avatar from "@material-ui/core/Avatar";
import List from "@material-ui/core/List";
import IconButton from "@material-ui/core/IconButton";
import Typography from "@material-ui/core/Typography";
import Divider from "@material-ui/core/Divider";
import ListItem from "@material-ui/core/ListItem";
import ListItemIcon from "@material-ui/core/ListItemIcon";
import ListItemText from "@material-ui/core/ListItemText";
import Box from "@material-ui/core/Box";
import useStyles from "./useStyles";
import globalStyles from "styles";
import menu from "./menu";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import mainLogo from "static/images/main_logo.png";
import smallLogo from "static/images/sml_logo.png";
import PratlipiLogo from "static/images/voicemail.png";
import logoutIcon from "static/images/logout_icon.png";
import { useDispatch } from "react-redux";

export const logoutChannel = new BroadcastChannel("logout");

const Sidebar = ({ open, setOpen }) => {
    const classes = useStyles();
    const dispatch = useDispatch();
    const history = useHistory();
    const globalClasses = globalStyles();
    const [anchorEl, setAnchorEl] = useState(null);
    const [userRoles, setUserRoles] = useState([]);
    const logoutMenuOpen = Boolean(anchorEl);

    const handleClick = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    const handleLogout = () => {
        logoutChannel.postMessage("Logout");
        history.push("/login");
        localStorage.clear();
        handleClose();
    };

    const openActionPopup = (index) => {
        dispatch({
            type: "SWITCH_COMPONENTS",
            payload: { selectedItem: index },
        });
    };

    const userRoleDetails = JSON.parse(localStorage.getItem("userRole"));

    useEffect(() => {
        const newArray = menu?.map((item, id) => {
            const keyValue = Object.keys(item)[0].toLowerCase();
            return {
                ...item,
                pageEnabled: userRoleDetails[keyValue],
            };
        });
        setUserRoles([...newArray]);
    }, []);

    return (
        <>
            <AppBar
                position="fixed"
                className={clsx(classes.appBar, {
                    [classes.appBarShift]: open,
                })}
            >
                <Toolbar
                    className={[globalClasses.flex, globalClasses.spaceBetween]}
                >
                    <Box
                        style={{
                            paddingLeft: "33%",
                            display: "flex",
                            alignItems: "center",
                        }}
                    >
                        <img
                            src={PratlipiLogo}
                            alt="Pratilipi MOM"
                            width="200px"
                            height="50px"
                        />
                        <Divider
                            className={classes.divider}
                            orientation="vertical"
                        />
                        <Typography variant="h5">Minutes of Meeting</Typography>
                    </Box>
                    <Box />
                    <Box display="flex">
                        <Box
                            display="flex"
                            flexDirection="column"
                            justifyContent="center"
                            mr={2}
                        >
                            <Typography
                                variant="body1"
                                className={globalClasses.mlAuto}
                            >
                                Welcome,
                            </Typography>
                            <Typography
                                variant="body1"
                                className={globalClasses.bold}
                            >
                                {localStorage.getItem("userName") || ""}
                            </Typography>
                        </Box>
                        <IconButton
                            color="inherit"
                            aria-label="Open drawer"
                            edge="start"
                            sixe="small"
                            onClick={handleClick}
                        >
                            <Avatar></Avatar>
                        </IconButton>
                        <Menu
                            id="basic-menu"
                            anchorEl={anchorEl}
                            open={logoutMenuOpen}
                            onClose={handleClose}
                            MenuListProps={{
                                "aria-labelledby": "basic-button",
                            }}
                            classes={{
                                root: classes.logoutDropdown,
                            }}
                        >
                            <MenuItem
                                onClick={handleLogout}
                                classes={{
                                    root: classes.logoutTextContainer,
                                }}
                            >
                                <img
                                    src={logoutIcon}
                                    alt="Logout"
                                    height="24px"
                                    width="21px"
                                    className={classes.logoutIconStyles}
                                />{" "}
                                <Typography
                                    display="inline"
                                    classes={{
                                        root: classes.logoutTextOne,
                                    }}
                                >
                                    Log
                                </Typography>
                                <Typography
                                    display="inline"
                                    classes={{
                                        root: classes.logoutText,
                                    }}
                                >
                                    out
                                </Typography>
                            </MenuItem>
                        </Menu>
                    </Box>
                </Toolbar>
            </AppBar>
            <Drawer
                variant="permanent"
                className={clsx(classes.drawer, {
                    [classes.drawerOpen]: open,
                    [classes.drawerClose]: !open,
                })}
                classes={{
                    paper: clsx({
                        [classes.drawerOpen]: open,
                        [classes.drawerClose]: !open,
                    }),
                }}
                onMouseEnter={() => setOpen(true)}
                onMouseLeave={() => setOpen(false)}
            >
                <Divider />
                <List>
                    <Box mb={8}>
                        <ListItem>
                            {open ? (
                                <img src={mainLogo} alt="" />
                            ) : (
                                <img src={smallLogo} alt="" />
                            )}
                        </ListItem>
                    </Box>
                    {userRoles.map(
                        (item, index) =>
                            // role[Object.keys(item)[0].toLowerCase()] && (
                            item.pageEnabled && (
                                <ListItem
                                    button
                                    key={index}
                                    component={NavLink}
                                    to={`/${Object.keys(
                                        item
                                    )[0].toLowerCase()}`}
                                    onClick={() => openActionPopup(index)}
                                    className={
                                        window.location.pathname.split(
                                            "/"
                                        )[1] ===
                                        Object.keys(item)[0].toLowerCase()
                                            ? classes.sidebarMenuSelected
                                            : classes.sidebarMenu
                                    }
                                >
                                    <ListItemIcon
                                        className={globalClasses.textWhite}
                                    >
                                        {item[Object.keys(item)[0]]}
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={Object.keys(item)[0]}
                                        className={globalClasses.textWhite}
                                    />
                                </ListItem>
                            )
                        // )
                    )}
                </List>
                <Divider style={{ background: "#ffff" }} />
            </Drawer>
        </>
    );
};

export default Sidebar;
