import React, { useEffect, useState } from "react";
import {
    Typography,
    Card,
    CardContent,
    Grid,
    Box,
    setRef,
} from "@material-ui/core";
import Checkbox from "@mui/material/Checkbox";
import globalSyles from "styles";
import useStyles from "screens/Dashboard/styles";
import { useDispatch, useSelector } from "react-redux";
import ActionsIcon from "static/Icons/actions-modified.svg";
import EscalationsIcon from "static/Icons/Escalations.svg";
import AnnouncementsIcon from "static/Icons/Announcements.png";
import AppreciationsIcon from "static/Icons/Appreciations.svg";
import RightArrowIcon from "static/Icons/rightArrow.svg";
import RangeDP from "components/RangeDP";
import { getSummary } from "store/action/summary";
import { format } from "date-fns";

const Insight = (props) => {
    const dispatch = useDispatch();
    const globalClasses = globalSyles();
    const classes = useStyles();
    const { summary } = useSelector((state) => state.summaryReducer);
    const [startDate, setStartDate] = useState(new Date());
    const [endDate, setEndDate] = useState(new Date());
    const images = [
        ActionsIcon,
        EscalationsIcon,
        AnnouncementsIcon,
        AppreciationsIcon,
    ];
    const { summaryUptoDate } = useState(false);
    const [checked, setChecked] = useState(false);
    const [isRange, setIsRange] = useState(false);
    var zero = 0;

    const handleStartDateChange = (data) => {
        setStartDate(new Date(data));
    };

    const handleEndDateChange = (data) => {
        setEndDate(new Date(data));
    };

    const handleDateChange = (data) => {
        if (data.length === 2) {
            setStartDate(new Date(data[0]));
            setEndDate(new Date(data[1]));
        } else {
            setStartDate(new Date(data));
            setEndDate(new Date(data));
        }
    };

    const handleChecked = () => {
        setChecked(!checked);
        //setIsRange(!isRange);
    };

    useEffect(() => {
        if (!summaryUptoDate) {
            const params = {
                start_date: format(startDate, "dd-MM-yyyy"),
                end_date: format(endDate, "dd-MM-yyyy"),
            };
            dispatch(getSummary(params));
        }
    }, [summaryUptoDate, startDate, endDate, dispatch]);

    const populateValue = (item) => {
        if (item === "Actions")
            return (
                <Box style={{ color: "#ff4d61" }}>
                    {summary?.dashboard_info?.action_count.toLocaleString(
                        "en-US",
                        {
                            minimumIntegerDigits: 2,
                            useGrouping: false,
                        }
                    ) ||
                        zero.toLocaleString("en-US", {
                            minimumIntegerDigits: 2,
                            useGrouping: false,
                        })}
                </Box>
            );
        else if (item === "Escalations")
            return (
                <Box style={{ color: "#3bb273" }}>
                    {summary?.dashboard_info?.escalation_count.toLocaleString(
                        "en-US",
                        {
                            minimumIntegerDigits: 2,
                            useGrouping: false,
                        }
                    ) ||
                        zero.toLocaleString("en-US", {
                            minimumIntegerDigits: 2,
                            useGrouping: false,
                        })}
                </Box>
            );
        else if (item === "Appreciations")
            return (
                <Box style={{ color: "#f2bc35" }}>
                    {summary?.dashboard_info?.appreciation_count.toLocaleString(
                        "en-US",
                        {
                            minimumIntegerDigits: 2,
                            useGrouping: false,
                        }
                    ) ||
                        zero.toLocaleString("en-US", {
                            minimumIntegerDigits: 2,
                            useGrouping: false,
                        })}
                </Box>
            );
        else if (item === "Announcements")
            return (
                <Box style={{ color: "#0b73aa" }}>
                    {summary?.dashboard_info?.announcement_count?.toLocaleString(
                        "en-US",
                        {
                            minimumIntegerDigits: 2,
                            useGrouping: false,
                        }
                    ) ||
                        zero.toLocaleString("en-US", {
                            minimumIntegerDigits: 2,
                            useGrouping: false,
                        })}
                </Box>
            );
    };

    const handelMoreBtn = (item) => {
        dispatch({
            type: "SELECTED_ACTION_ROW_DATA",
            payload: {
                selectedCardInfo: { cardTitle: item, openCardModal: true },
            },
        });
    };

    return (
        <>
            <Grid xs={12} item className={classes.titleBar}>
                <Grid xs={checked ? 7 : 10} item>
                    <Typography
                        align="left"
                        className={`${classes.insights} ${globalClasses.bold}`}
                    >
                        Insights
                    </Typography>
                </Grid>
                {checked ? (
                    <Grid xs={2.5} item>
                        <Box
                            style={{
                                textAlign: "left",
                                fontFamily: "Lato",
                                fontSize: "14px",
                                fontWeight: "bold",
                                color: "#333333",
                                paddingBottom: "10px",
                            }}
                        >
                            Select Start Date
                        </Box>
                        <RangeDP
                            handleDateChange={handleStartDateChange}
                            isRange={isRange}
                        />
                    </Grid>
                ) : (
                    ""
                )}
                {checked ? (
                    <Grid xs={2.5} item>
                        <Box
                            style={{
                                textAlign: "left",
                                fontFamily: "Lato",
                                fontSize: "14px",
                                fontWeight: "bold",
                                color: "#333333",
                                paddingBottom: "10px",
                            }}
                        >
                            Select End Date
                        </Box>
                        <RangeDP
                            handleDateChange={handleEndDateChange}
                            isRange={isRange}
                        />
                        <Box
                            style={{
                                textAlign: "left",
                                fontFamily: "Lato",
                                fontSize: "14px",
                                color: "#333333",
                            }}
                        >
                            <Checkbox
                                checked={checked}
                                onChange={handleChecked}
                                name="checkedB"
                            />
                            Choose custom range
                        </Box>
                    </Grid>
                ) : (
                    <Grid xs={2.5} item>
                        <Box
                            style={{
                                textAlign: "left",
                                fontFamily: "Lato",
                                fontSize: "14px",
                                fontWeight: "bold",
                                color: "#333333",
                                paddingBottom: "10px",
                            }}
                        >
                            Select a Date
                        </Box>
                        <RangeDP
                            handleDateChange={handleDateChange}
                            isRange={isRange}
                        />
                        <Box
                            style={{
                                textAlign: "left",
                                fontFamily: "Lato",
                                fontSize: "14px",
                                color: "#333333",
                            }}
                        >
                            <Checkbox
                                checked={checked}
                                onChange={handleChecked}
                                name="checkedB"
                            />
                            Choose custom range
                        </Box>
                    </Grid>
                )}
            </Grid>
            <Grid container spacing={24} style={{ paddingLeft: "24.7px" }}>
                {[
                    "Actions",
                    "Escalations",
                    "Announcements",
                    "Appreciations",
                ].map((item, idx) => (
                    <Grid item md={3}>
                        <Card className={classes.dashboardListCard}>
                            <Grid
                                style={{
                                    display: "flex",
                                    alignItems: "center",
                                }}
                            >
                                <Grid>
                                    <img
                                        src={images[idx]}
                                        alt=""
                                        width="22px"
                                        height="23px"
                                        margin="10px"
                                    />
                                </Grid>
                                <Grid>
                                    <Typography
                                        align="left"
                                        className={`${classes.insightsBody} ${globalClasses.bold} ${classes.titleText}`}
                                    >
                                        {item}
                                    </Typography>
                                </Grid>
                            </Grid>
                            <CardContent style={{ height: "165px" }}>
                                <Grid className={classes.innerCardTitle}>
                                    {populateValue(item)}
                                </Grid>
                                <Grid className={classes.innerCardSubTitle}>
                                    {"Total " + item}
                                </Grid>
                            </CardContent>
                            <Grid
                                style={{
                                    display: "flex",
                                    alignItems: "baseline",
                                    justifyContent: "flex-end",
                                    cursor: "pointer",
                                }}
                                className={classes.moreHover}
                                onClick={() => handelMoreBtn(item)}
                            >
                                <Grid className={classes.more}>{"More"}</Grid>
                                <img
                                    src={RightArrowIcon}
                                    alt=""
                                    width="12px"
                                    height="9px"
                                    margin="10px"
                                    className={classes.more}
                                />
                            </Grid>
                        </Card>
                    </Grid>
                ))}
            </Grid>
        </>
    );
};

export default Insight;
