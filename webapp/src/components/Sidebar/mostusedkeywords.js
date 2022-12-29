import React, { useEffect } from "react";
import Divider from "@material-ui/core/Divider";
import Collapse from "@material-ui/core/Collapse";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";
import IconExpandLess from "@material-ui/icons/ExpandLess";
import IconExpandMore from "@material-ui/icons/ExpandMore";
import { withStyles } from "@material-ui/core/styles";
import Badge from "@material-ui/core/Badge";

import globalStyles from "styles";
import { Typography } from "@material-ui/core";

const StyledBadge = withStyles((theme) => ({
    badge: {
        border: `2px solid ${theme.palette.common.white}`,
        background: theme.palette.common.white,
        color: theme.palette.primary.main,
    },
}))(Badge);

const generateMostUsedKeywords = (arr) => {
    const jsx = [];
    for (let i = 0; i < arr.length; i++) {
        if (i === 3) {
            jsx.push(
                <ListItemText style={{ marginTop: "10px" }}>
                    <Typography
                        variant="span"
                        style={{ textDecoration: "underline" }}
                    >
                        View all
                    </Typography>
                </ListItemText>
            );
            break;
        }
        jsx.push(
            <ListItemText style={{ marginTop: "10px" }}>
                <Typography variant="span">{Object.keys(arr[i])[0]}</Typography>
                <StyledBadge
                    badgeContent={arr[i][Object.keys(arr[i])[0]]}
                    style={{ marginLeft: "18px" }}
                />
            </ListItemText>
        );
    }
    return jsx;
};

const MostUsedKeywords = ({ sideBarOpen }) => {
    const globalClasses = globalStyles();
    const items = [
        // {
        //   People: [
        //     { Willian: 17 },
        //     { Anderson: 24 },
        //     { Robertson: 5 },
        //     { Robertson: 5 },
        //     { Robertson: 5 },
        //   ],
        // },
        // { Applications: [{ Figma: 17 }, { Jira: 4 }, { Vscode: 8 }, { Pycharm: 8 }] },
        // { Topic: [{ 'AI/ML': 17 }, { Minutesofmeeting: 4 }, { MoMgeneration: 8 }] },
    ];

    const [open, setOpen] = React.useState(false);

    useEffect(() => {
        setOpen(sideBarOpen);
    }, [sideBarOpen]);

    return (
        <List>
            <ListItem
                button
                className={globalClasses.textWhite}
                onClick={() => setOpen(!open)}
            >
                {sideBarOpen && <ListItemText primary={"Most used keywords"} />}
                {!open && <IconExpandMore />}
                {open && <IconExpandLess />}
            </ListItem>
            <Collapse in={open} timeout="auto" unmountOnExit>
                <Divider />
                <List component="div" disablePadding>
                    {items.length > 0 ? (
                        items.map((item) => (
                            <>
                                <ListItem style={{ paddingLeft: "30px" }}>
                                    <ListItemText
                                        className={globalClasses.textWhite}
                                    >
                                        <Typography
                                            variant="span"
                                            style={{ fontWeight: "600" }}
                                        >
                                            {`${Object.keys(item)[0]}  (${
                                                Object.values(item)[0].length
                                            })`}
                                        </Typography>
                                        {generateMostUsedKeywords(
                                            Object.values(item)[0]
                                        )}
                                    </ListItemText>
                                </ListItem>
                                <Divider
                                    style={{
                                        background: "rgba(255,255,255,0.3)",
                                    }}
                                />
                            </>
                        ))
                    ) : (
                        <Typography
                            className={globalClasses.textWhite}
                            style={{ marginTop: "50px" }}
                        >
                            No keywords to display!
                        </Typography>
                    )}
                </List>
            </Collapse>
        </List>
    );
};

export default MostUsedKeywords;
