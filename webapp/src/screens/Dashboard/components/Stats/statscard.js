import React from "react";
import { Typography, Box, Card } from "@material-ui/core";
import Avatar from "@material-ui/core/Avatar";
import useStyles from "screens/Dashboard/styles";

const StatsCard = ({ icon, count, footer, color }) => {
    const classes = useStyles();
    return (
        <Card className={classes.statsCard}>
            <Box
                display="flex"
                style={{
                    alignItems: "center",
                }}
            >
                <Avatar
                    style={{
                        backgroundColor: color,
                        width: "30px",
                        height: "30px",
                        margin: "0 8px 8px 8px",
                        padding: "14px 13px 11px 12px",
                    }}
                >
                    <img src={icon} alt="" className={classes.buttonIcon}></img>
                </Avatar>
                <Box alignContent="left" pl={2}>
                    <Box
                        style={{
                            color: color,
                            fontSize: "24px",
                            fontWeight: "bold",
                            textAlign: "left",
                        }}
                    >
                        {count}
                    </Box>
                    <Box
                        style={{
                            color: "#333333",
                            fontSize: "16px",
                            fontWeight: "normal",
                            textAlign: "left",
                            paddingTop: "4px",
                            fontFamily: "Lato",
                        }}
                    >
                        {footer}
                    </Box>
                </Box>
            </Box>
        </Card>
    );
};

export default StatsCard;
