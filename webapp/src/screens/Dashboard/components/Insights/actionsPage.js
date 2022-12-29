import React, { useState } from "react";
import "react-date-range/dist/styles.css"; // main css file
import "react-date-range/dist/theme/default.css"; // theme css file
import useStyles from "./screens/Dashboard/styles";
import {
    Box,
    Typography,
    Divider,
    Paper,
    Grid,
    Menu,
    MenuItem,
    Button,
    Checkbox,
} from "@material-ui/core";
import Table from "components/Table";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import { useDispatch, useSelector } from "react-redux";
import useSumaryStyles from "screens/FeedBackLoop/components/Summary/styles";
import customStyles from "screens/FeedBackLoop/components/MoMView/useStyles";
import { DateRangePicker } from "react-date-range";
import { addDays } from "date-fns";
import { data } from "./actionJsonData";

import KeyboardArrowRightIcon from "@mui/icons-material/KeyboardArrowRight";
import RecommendedTalktime from "./actionsGraph";
import { useHistory } from "react-router-dom";
import Event from "@mui/icons-material/Event";
import Sort from "@mui/icons-material/Sort";
import BarGraph from "./BarGraph";

let inputData = {
    dataKey: "hour",
    oyLabel: "Total Count",
    oxLabel: "hours",
    yLimit: [0, 20000],
    values: [
        { hour: 0, Team1: 4000, Team2: 2400, Team3: 2120 },
        { hour: 1, Team1: 3000, Team2: 1398 },
        { hour: 2, Team1: 2000, Team2: 9800, Team3: 3220 },
        { hour: 3, Team1: 2780, Team2: 3908 },
        { hour: 4, Team1: 1890, Team2: 4800, Team3: 1220 },
        { hour: 5, Team1: 2390, Team2: 3800 },
        { hour: 6, Team1: 3490, Team2: 4300 },
        { hour: 7, Team1: 4000, Team2: 2400, Team3: 2120 },
        { hour: 9, Team1: 3000, Team2: 1398 },
        { hour: 10, Team1: 2000, Team2: 9800, Team3: 3220 },
        { hour: 11, Team1: 2780, Team2: 3908 },
        { hour: 12, Team1: 1890, Team2: 4800, Team3: 1220 },
        { hour: 13, Team1: 2390, Team2: 3800 },
        { hour: 14, Team1: 3490, Team2: 4300 },
        { hour: 15, Team1: 1890, Team2: 4800, Team3: 1220 },
        { hour: 16, Team1: 2390, Team2: 3800 },
        { hour: 17, Team1: 3490, Team2: 4300 },
        { hour: 18, Team1: 4000, Team2: 2400, Team3: 2120 },
        { hour: 19, Team1: 3000, Team2: 1398 },
        { hour: 20, Team1: 2000, Team2: 9800, Team3: 3220 },
        { hour: 21, Team1: 2780, Team2: 3908 },
        { hour: 22, Team1: 1890, Team2: 4800, Team3: 1220 },
        { hour: 23, Team1: 2390, Team2: 3800 },
    ],
};

let inputLabels = [
    { key: "Team1", color: "#8884d8" },
    { key: "Team2", color: "#82ca9d" },
    { key: "Team3", color: "#81cc2d" },
];

const MyNewTitle = ({ text, variant }) => (
    <Typography
        variant={variant}
        style={{
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            fontWeight: "bold",
        }}
    >
        {text}
    </Typography>
);

let dropDownOptions = [
    { value: "Team1", label: "Team1" },
    { value: "Team2", label: "Team2" },
    { value: "Team3", label: "Team3" },
    { value: "Team4", label: "Team4" },
    { value: "Team5", label: "Team5" },
];

const ActionsComp = () => {
    const classes = useStyles();
    const momCls = customStyles();
    let history = useHistory();
    const summaryClasses = useSumaryStyles();
    const dispatch = useDispatch();
    const { selectedActionRowData, selectedActionRowIdx } = useSelector(
        (state) => state.dashboardActionTabReducer
    );
    const { selectedIdx: storeSelectedIdx, selectedTab } = useSelector(
        (state) => state.insightsReducer
    );

    const [state, setState] = useState([
        {
            startDate: new Date(),
            endDate: addDays(new Date(), 7),
            key: "selection",
        },
    ]);

    const [anchorEl, setAnchorEl] = useState(null);
    const openDatePicker = Boolean(anchorEl);

    const [anchorSectionEl, setAnchorSectionEl] = useState(null);
    const [checked, setChecked] = useState({});
    const sectionopen = Boolean(anchorSectionEl);
    const [selectedTeamsData, setSelectedTeams] = useState({});
    const [opacity, setOpacity] = React.useState({
        uv: 1,
        pv: 1,
        amt: 1,
    });

    const handleClose = () => {
        setAnchorEl(null);
    };

    const handleDatePickerClick = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const openTeamsPopup = (event) => {
        setAnchorSectionEl(event.currentTarget);
    };
    const closeTeamsPopup = () => {
        setAnchorSectionEl(null);
    };

    const seletedSections = (value, idx) => {
        if (checked[idx]) {
            setChecked({ ...checked, [idx]: false });
        } else {
            setChecked({ ...checked, [idx]: true });
        }
        setSelectedTeams({
            [value]: [],
        });
    };

    const addTeams = () => {
        setAnchorSectionEl(null);
    };

    const columns = [
        {
            title: "S.NO",
            field: "id",
            width: "15%",
            align: "center",
            headerStyle: {
                borderBottom: "2px solid #F2F2F2",
            },
            cellStyle: {
                borderBottom: "2px solid #F2F2F2",
                fontWeight: "600",
                paddingRight: "43px",
            },
        },

        {
            title: "Team Name",
            field: "teamname",
            width: "15%",
            align: "center",
            headerStyle: {
                borderBottom: "2px solid #F2F2F2",
            },
            cellStyle: {
                borderBottom: "2px solid #F2F2F2",
                paddingRight: "43px",
            },
        },
        {
            title: "Total Action Items",
            field: "totalactionitems",
            width: "15%",
            align: "center",
            headerStyle: {
                borderBottom: "2px solid #F2F2F2",
            },
            cellStyle: {
                borderBottom: "2px solid #F2F2F2",
                paddingRight: "43px",
            },
        },
        {
            title: "Completed",
            field: "completed",
            width: "15%",
            align: "center",
            headerStyle: {
                borderBottom: "2px solid #F2F2F2",
            },
            cellStyle: {
                borderBottom: "2px solid #F2F2F2",
                paddingRight: "43px",
            },
        },
        {
            title: "Upcoming",
            field: "upcoming",
            width: "15%",
            align: "center",
            headerStyle: {
                borderBottom: "2px solid #F2F2F2",
            },
            cellStyle: {
                borderBottom: "2px solid #F2F2F2",
                paddingRight: "43px",
            },
        },
        {
            title: "Performance",
            field: "talktime",
            width: "20%",
            align: "center",
            headerStyle: {
                borderBottom: "2px solid #F2F2F2",
            },
            cellStyle: {
                borderBottom: "2px solid #F2F2F2",
                paddingLeft: "43px",
                width: "20%",
            },
            render: (props) => <RecommendedTalktime data={props.talktime} />,
        },
        {
            title: "",
            field: "action",
            width: "10%",
            align: "center",
            headerStyle: {
                borderBottom: "2px solid #F2F2F2",
            },
            cellStyle: {
                borderBottom: "2px solid #F2F2F2",
                paddingLeft: "43px",
                width: "10%",
            },
            render: (rowData) => (
                <Box fontSize={16} fontWeight="fontWeightMedium">
                    <KeyboardArrowRightIcon
                        onClick={() => openActionPopup(rowData)}
                    />
                </Box>
            ),
        },
    ];

    const openActionPopup = (rowData) => {
        dispatch({
            type: "SELECTED_ACTION_ROW_DATA",
            payload: {
                selectedActionRowData: rowData,
                selectedActionRowIdx: 1,
            },
        });
        dispatch({
            type: "SWITCH_INSIGHT_TABS",
            payload: { selectedTab: "Escalations", selectedIdx: 3 },
        });
    };

    const handleClick = () => {};
    const handleMouseEnter = (o) => {
        const { dataKey } = o;
        setOpacity({ ...opacity, [dataKey]: 0.5 });
    };
    const handleMouseLeave = (o) => {
        const { dataKey } = o;
        // setOpacity({ ...opacity, [dataKey]: 1 });
    };

    return (
        <Grid container item xs={12} direction="column">
            <Grid container spacing={2}>
                <Grid container item xs={8} direction="column">
                    <Typography className={classes.fontCss}>
                        {selectedTab} Overview
                    </Typography>
                </Grid>
                <Grid
                    container
                    item
                    xs={2}
                    direction="column"
                    className={momCls.teambtnwidth}
                >
                    <Button
                        variant="outlined"
                        className={momCls.buttonClsTeam}
                        onClick={openTeamsPopup}
                    >
                        <Sort style={{ marginRight: "6px" }} /> Select Team
                    </Button>
                    <Menu
                        id="basic-menu"
                        anchorEl={anchorSectionEl}
                        open={sectionopen}
                        onClose={closeTeamsPopup}
                        MenuListProps={{
                            "aria-labelledby": "basic-button",
                        }}
                        style={{ top: "45px" }}
                        className={momCls.menuWidth}
                    >
                        <Typography
                            className={momCls.labelCls}
                            style={{ float: "none" }}
                        >
                            {"Teams"}
                        </Typography>
                        <Box
                            component="div"
                            display="block"
                            className={momCls.listScrollTeams}
                        >
                            {dropDownOptions.map((item, idx) => {
                                return (
                                    <Box component="div" display="flex">
                                        <Checkbox
                                            checked={checked[idx]}
                                            inputProps={{
                                                "aria-label": "controlled",
                                            }}
                                            onChange={() =>
                                                seletedSections(item, idx)
                                            }
                                            className={momCls.checkboxcolor}
                                        />
                                        <MenuItem>{item.value}</MenuItem>
                                    </Box>
                                );
                            })}
                        </Box>
                        <Box
                            component="div"
                            display="flex"
                            className={momCls.btngroup}
                        >
                            <Button
                                variant="text"
                                onClick={closeTeamsPopup}
                                className={momCls.btnMargin}
                            >
                                cancel
                            </Button>
                            <Button
                                variant="contained"
                                color="primary"
                                className={momCls.btnMargin}
                                onClick={addTeams}
                            >
                                Apply
                            </Button>
                        </Box>
                    </Menu>
                </Grid>
                <Grid container item xs={2} direction="column">
                    <Box component="div" className={classes.sortcls}>
                        <Button
                            id="basic-button"
                            aria-controls="basic-menu"
                            aria-haspopup="true"
                            aria-expanded={openDatePicker ? "true" : undefined}
                            style={
                                openDatePicker
                                    ? { background: "lightgray" }
                                    : { background: "" }
                            }
                            onClick={handleDatePickerClick}
                            className={`${classes.transformvalue}`}
                        >
                            <Event style={{ marginRight: "6px" }} /> Select Date
                            <KeyboardArrowDownIcon
                                style={{ marginLeft: "6px" }}
                            />
                        </Button>
                        <Menu
                            id="basic-menu"
                            anchorEl={anchorEl}
                            open={openDatePicker}
                            onClose={handleClose}
                            MenuListProps={{
                                "aria-labelledby": "basic-button",
                            }}
                            anchorOrigin={{
                                vertical: "bottom",
                                horizontal: "right",
                            }}
                            transformOrigin={{
                                vertical: "top",
                                horizontal: "right",
                            }}
                            style={{ top: "24%" }}
                        >
                            <DateRangePicker
                                onChange={(item) => setState([item.selection])}
                                showSelectionPreview={true}
                                moveRangeOnFirstSelection={false}
                                months={2}
                                ranges={state}
                                direction="horizontal"
                            />
                            <MenuItem className={classes.btnCls}>
                                <Button color={"#d8d8d8"} onClick={handleClose}>
                                    Cancel
                                </Button>
                                <Button
                                    variant="contained"
                                    className={classes.btnColor}
                                    onClick={handleClose}
                                >
                                    Apply
                                </Button>
                            </MenuItem>
                        </Menu>
                    </Box>
                </Grid>
            </Grid>
            <Divider className={classes.dividerCss} />

            <Paper className={classes.cards}>
                <BarGraph
                    title="DayAhead Predicted Consumption"
                    data={inputData}
                    labels={inputLabels}
                />
                <Box mt={15}>
                    <Table
                        data={data}
                        columns={columns}
                        title={
                            <MyNewTitle
                                variant="h5"
                                text="Teamwise performance"
                            />
                        }
                        from={"actionTab"}
                    />
                </Box>
            </Paper>
        </Grid>
    );
};
export default ActionsComp;
