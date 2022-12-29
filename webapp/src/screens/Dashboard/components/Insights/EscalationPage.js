import React, { useState, useEffect } from "react";
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
} from "@material-ui/core";
import Sort from "@mui/icons-material/Event";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import { useDispatch, useSelector } from "react-redux";
import useSumaryStyles from "screens/FeedBackLoop/components/Summary/styles";
import ReactECharts from "echarts-for-react";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import LinearProgress, {
    linearProgressClasses,
} from "@mui/material/LinearProgress";
import customStyles from "screens/FeedBackLoop/components/MoMView/useStyles";
import Select from "react-select";
import Modal from "components/Modal";
import {
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Collapse,
} from "@mui/material";
import videoIcon from "static/images/video.svg";
import PreviewFile from "screens/FeedBackLoop/components/Preview";
import { DateRangePicker } from "react-date-range";
import { addDays } from "date-fns";
import {
    RemoveCircleOutline,
    ExpandLess,
    ExpandMore,
    AddCircleOutline,
} from "@mui/icons-material";

let result = [
    {
        value: 335,
        name: "TrailBazzer",
    },
    {
        value: 310,
        name: "Sense MAker",
    },
    { value: 234, name: "Hive" },
    {
        value: 135,
        name: "Death Eaters",
    },
    {
        value: 1548,
        name: "Transcribers",
    },
    {
        value: 1548,
        name: "Hive Team",
    },
    {
        value: 1548,
        name: "our team",
    },
    {
        value: 1548,
        name: "Friends team",
    },
    {
        value: 1548,
        name: "Asm Team",
    },
    {
        value: 1548,
        name: "BigBoss team",
    },
];

let dropDownOptions = [
    { value: "Today", label: "Today" },
    { value: "Yesterday", label: "Yesterday" },
    { value: "Last week", label: "Last week" },
    { value: "Last month", label: "Last month" },
    { value: "Last Year", label: "Last year" },
];
let colorPalette = [
    "#00b04f",
    "#ffbf00",
    "#ff7f50",
    "#87cefa",
    "#da70d6",
    "#32cd32",
    "#6495ed",
    "#ff69b4",
    "#ba55d3",
    "#cd5c5c",
    "#ffa500",
    "#40e0d0",
];
const fileData = {
    name: "HIVE_Video_80204842_2439022",
    fullname: "HIVE_Video_80204842_2439022.mp4",
    teamName: "HIVE",
    type: ".mp4",
    date: "30 Jul 2021",
    status: "Completed",
    size: "32265630",
    src: "",
};
const EscalationsComp = () => {
    const classes = useStyles();
    const momCls = customStyles();
    const dispatch = useDispatch();
    const summaryClasses = useSumaryStyles();
    const [selectedValue, setSelectedValue] = useState();
    const [popupcontent, setOpenPopupContent] = useState({});
    const [isOpenPopup, setOpenPopup] = useState(false);
    const [open, setOpen] = useState({});
    const [selectedIdx, setSelectedIdx] = useState({});
    const [innerSelectedIdx, setInnerSelectedIdx] = useState({});
    const { momJson } = useSelector((state) => state.momReducer);
    let MomEntries = momJson?.mom_entries || {};
    const [getData, setGetData] = useState(MomEntries);
    const [openPreview, setOpenPreview] = useState(false);
    const [fileName, setFileName] = useState("");
    const [fileSize, setFileSize] = useState("");
    const [fileSource, setFileSource] = useState("");
    const [ParentSelectedIdx, setParentSelectedIdx] = useState({});
    const [SelectedCard, setSelectedCard] = useState();
    const [SelectedTeam, setSelectedTeam] = useState();
    const [innerIdxArr, setInnerIdxArr] = useState({});
    const { selectedIdx: storeSelectedIdx, selectedTab } = useSelector(
        (state) => state.insightsReducer
    );
    const { selectedActionRowData, selectedActionRowIdx } = useSelector(
        (state) => state.dashboardActionTabReducer
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

    useEffect(() => {
        if (selectedActionRowIdx) {
            setSelectedTeam(result[0]);
            openItemsPopup("Contributions", 2);
        }
    }, []);

    const openItemsPopup = (val, card) => {
        let title =
            val === "Recent"
                ? val + " " + selectedTab
                : val === "Contributions"
                ? selectedTab.slice(0, -1) + " " + val
                : val === "Most " && selectedTab === "Escalations"
                ? "Most Escalated Participants"
                : "Most Appreciated Participants";

        setOpenPopupContent({ ...popupcontent, title });
        setOpenPopup(true);
        setSelectedCard(card);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    const handleDatePickerClick = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const handleModalClose = () => {
        if (selectedActionRowIdx !== "") {
            dispatch({
                type: "SELECTED_ACTION_ROW_DATA",
                payload: {
                    selectedActionRowData: {},
                    selectedActionRowIdx: "",
                },
            });
        }
        setOpenPopup(false);
        setInnerSelectedIdx({});
        setSelectedIdx({});
    };

    const handlePreviewOpen = (rowData) => {
        setOpenPreview(true);
        setFileName(rowData.fullname);
        setFileSize(rowData.size);
        setFileSource(rowData.src);
    };

    const handlePreviewClose = (e) => {
        setOpenPreview(false);
    };

    const Doughnut = () => {
        return {
            title: {
                top: 25,
                left: 20,
                text: "Contribution",
            },
            tooltip: {
                trigger: "item",
                formatter: "{a} <br/>{b}: {c} ({d}%)",
            },
            series: [
                {
                    color: colorPalette,
                    name: "Contribution",
                    type: "pie",
                    radius: ["70%", "90%"],
                    center: ["50%", "50%"],
                    top: 30,
                    //   startAngle: 180,
                    //   endAngle: 360,
                    avoidLabelOverlap: true,
                    label: {
                        show: true,
                        position: "center",
                        fontWeight: "bold",
                        formatter: `${"192"} \n ${selectedTab}`,
                    },
                    emphasis: {
                        label: {
                            show: true,
                            fontSize: "16",
                            fontWeight: "bold",
                        },
                    },
                    labelLine: {
                        show: true,
                        text: "Total",
                    },
                    data: result,
                },
            ],
        };
    };

    const onChange = (e) => {
        setSelectedValue(e);
    };

    const handleClick = (idx) => {
        if (selectedIdx[idx] === idx) {
            delete selectedIdx[idx];
            setSelectedIdx(selectedIdx);
            setOpen({ ...open, [idx]: false });
        } else {
            setSelectedIdx({ ...selectedIdx, [idx]: idx });
            setOpen({ ...open, [idx]: true });
        }
    };

    const handleInnerClick = (idx, parentIdx) => {
        if (
            // innerSelectedIdx[idx] === idx
            innerIdxArr[parentIdx]?.length > 0 &&
            innerIdxArr[parentIdx].includes(idx)
        ) {
            // delete innerSelectedIdx[idx];
            // setInnerSelectedIdx(innerSelectedIdx);
            innerIdxArr[parentIdx] = innerIdxArr[parentIdx].filter(
                (item) => item !== idx
            );
            setInnerIdxArr(innerIdxArr);
            // setOpen({ ...open, [idx]: false });
        } else {
            let arr =
                innerIdxArr[parentIdx]?.length > 0
                    ? innerIdxArr[parentIdx]
                    : [];
            arr.push(idx);
            setInnerIdxArr({ ...innerIdxArr, [parentIdx]: arr });
            // setInnerSelectedIdx({ ...innerSelectedIdx, [idx]: idx });
            // setOpen({ ...open, [idx]: true });
        }

        if (ParentSelectedIdx[parentIdx] === parentIdx) {
            setParentSelectedIdx({
                ...ParentSelectedIdx,
                [parentIdx]: parentIdx,
            });
        } else {
            setParentSelectedIdx({
                ...ParentSelectedIdx,
                [parentIdx]: parentIdx,
            });
        }
    };

    const InnerListItemIteration = (props) => {
        let { innerdoc, innerindex, innertitle, parentIdx } = props;
        innerdoc = innerdoc.map((item, idx) => {
            return { ...item, id: idx + 1 };
        });
        return (
            <>
                <ListItem
                    button
                    className={classes.collapsecls}
                    key={innerindex}
                >
                    <Box component="div" display="inline" fontWeight={"500"}>
                        {parentIdx === ParentSelectedIdx[parentIdx] &&
                        innerindex === innerSelectedIdx[innerindex] ? (
                            // innerIdxArr[parentIdx]?.length > 0 &&
                            // innerIdxArr[parentIdx].includes(innerindex)
                            <RemoveCircleOutline
                                onClick={() =>
                                    handleInnerClick(innerindex, parentIdx)
                                }
                                className={classes.removeIcon}
                            />
                        ) : (
                            <AddCircleOutline
                                onClick={() =>
                                    handleInnerClick(innerindex, parentIdx)
                                }
                                className={classes.removeIcon}
                            />
                        )}
                    </Box>
                    <ListItemText
                        primary={innertitle}
                        className={` ${classes.titlebarinner}`}
                    />
                </ListItem>
                <Collapse
                    key={Math.random()}
                    in={open[innerindex]}
                    timeout="auto"
                    unmountOnExit
                >
                    <List component="li" disablePadding key={Math.random()}>
                        {
                            // parentIdx === ParentSelectedIdx[parentIdx] &&
                            //   innerindex === innerSelectedIdx[innerindex] &&
                            innerIdxArr[parentIdx]?.includes(innerindex) && (
                                <div
                                    style={{ maxWidth: "96%" }}
                                    key={Math.random()}
                                >
                                    {[1, 2, 3, 4, 6, 7, 8, 9, 0].map((item) => (
                                        <Grid
                                            container
                                            alignItems="center"
                                            spacing={1}
                                            className={
                                                classes.titlebarinnerTitle
                                            }
                                        >
                                            <Grid
                                                item
                                                xs={1}
                                                className={
                                                    classes.profileBarVideoIcon
                                                }
                                            >
                                                <Box
                                                    component="div"
                                                    display="inline"
                                                    m={1}
                                                    fontWeight={"500"}
                                                >
                                                    <img
                                                        src={videoIcon}
                                                        height="20px"
                                                        width="20px"
                                                        className={
                                                            classes.videoIcon
                                                        }
                                                        data-tip="Play Files"
                                                        onClick={() =>
                                                            handlePreviewOpen(
                                                                fileData
                                                            )
                                                        }
                                                        alt=""
                                                    />
                                                </Box>
                                            </Grid>
                                            <Grid
                                                item
                                                xs={1}
                                                className={
                                                    classes.profileBarDate
                                                }
                                            >
                                                <Box
                                                    component="div"
                                                    display="inline"
                                                    m={1}
                                                    mr={0}
                                                    ml={-1}
                                                    fontWeight={"500"}
                                                >
                                                    Aug 12
                                                </Box>
                                            </Grid>
                                            <Grid item xs={10}>
                                                <Box
                                                    component="div"
                                                    display="inline"
                                                    m={1}
                                                    mr={0}
                                                    ml={-1}
                                                    fontWeight={"500"}
                                                >
                                                    Participants and their{" "}
                                                    {selectedTab} Participants
                                                    and their {selectedTab}{" "}
                                                    Participants and their{" "}
                                                    {selectedTab}
                                                </Box>
                                            </Grid>
                                        </Grid>
                                    ))}
                                </div>
                            )
                        }
                    </List>
                </Collapse>
            </>
        );
    };

    const CustomizedListItem = (props) => {
        let { doc, index, title } = props;
        doc = doc.map((item, idx) => {
            return { ...item, id: idx + 1 };
        });
        return (
            <>
                <ListItem button className={classes.collapseclsCss} key={index}>
                    <Typography className={classes.collapseSno}>
                        {index.toString().padStart(2, "0")}
                    </Typography>
                    <ListItemText
                        primary={title}
                        className={`${momCls.collapse_name} ${classes.collapseTitleCss}`}
                    />

                    {index === selectedIdx[index] ? (
                        <ExpandLess onClick={() => handleClick(index)} />
                    ) : (
                        <ExpandMore onClick={() => handleClick(index)} />
                    )}
                </ListItem>
                <Collapse
                    key={Math.random()}
                    in={open[index]}
                    timeout="auto"
                    unmountOnExit
                >
                    <List component="li" disablePadding key={Math.random()}>
                        {index === selectedIdx[index] && SelectedCard === 1 ? (
                            <InnerItemsRepeat index={index} />
                        ) : (
                            index === selectedIdx[index] && (
                                <InnerItemsList index={index} />
                            )
                        )}
                    </List>
                </Collapse>
            </>
        );
    };

    const InnerItemsRepeat = (props) => {
        let { index } = props;
        return (
            <div style={{ maxWidth: "98%" }} key={Math.random()}>
                <Grid container className={`${classes.collapseScrollinner}`}>
                    <Grid item xs={12} direction="column">
                        <List
                            component="nav"
                            aria-labelledby="nested-list-subheader"
                        >
                            {Object.keys(getData).map((element, InnerIndex) => {
                                return (
                                    <InnerListItemIteration
                                        innerindex={InnerIndex}
                                        innerdoc={getData[element]}
                                        innertitle={element}
                                        parentIdx={index}
                                    />
                                );
                            })}
                        </List>
                    </Grid>
                </Grid>
            </div>
        );
    };

    const InnerItemsList = (props) => {
        let { index } = props;
        return (
            <div
                style={{ maxWidth: "96%" }}
                key={Math.random()}
                className={classes.firstBarCss}
            >
                {[1, 2, 3, 4, 6, 7, 8, 9, 0].map((item) => (
                    <Grid
                        container
                        alignItems="center"
                        spacing={1}
                        className={classes.titlebarinnerTitle}
                    >
                        <Grid
                            item
                            xs={1}
                            className={classes.profileBarVideoIcon}
                        >
                            <Box
                                component="div"
                                display="inline"
                                m={1}
                                fontWeight={"500"}
                            >
                                <img
                                    src={videoIcon}
                                    height="20px"
                                    width="20px"
                                    className={classes.videoIcon}
                                    data-tip="Play Files"
                                    onClick={() => handlePreviewOpen(fileData)}
                                    alt=""
                                />
                            </Box>
                        </Grid>
                        <Grid item xs={1} className={classes.profileBarDate}>
                            <Box
                                component="div"
                                display="inline"
                                m={1}
                                mr={0}
                                ml={-1}
                                fontWeight={"500"}
                            >
                                Aug 12
                            </Box>
                        </Grid>
                        <Grid item xs={10}>
                            <Box
                                component="div"
                                display="inline"
                                m={1}
                                mr={0}
                                ml={-1}
                                fontWeight={"500"}
                            >
                                Participants and their {selectedTab}{" "}
                                Participants and their {selectedTab}{" "}
                                Participants and their {selectedTab}
                            </Box>
                        </Grid>
                    </Grid>
                ))}
            </div>
        );
    };

    const popUPContent = () => {
        return (
            <>
                {SelectedCard === 2 && (
                    <Grid
                        container
                        alignItems="center"
                        className={classes.firstBarCss}
                    >
                        <Box component="div" display="inline" m={2} mr={1}>
                            Team name:
                        </Box>
                        <Box
                            component="div"
                            display="inline"
                            fontWeight={"bold"}
                            m={2}
                            mr={6}
                        >
                            {SelectedTeam && SelectedTeam["name"]}
                        </Box>
                        <Divider
                            orientation="vertical"
                            flexItem
                            className={classes.dividerStyle}
                        />
                        <Box component="div" display="inline" m={2} mr={1}>
                            Team name:
                        </Box>
                        <Box
                            component="div"
                            display="inline"
                            fontWeight={"bold"}
                            m={2}
                            mr={6}
                        >
                            {SelectedTeam && SelectedTeam["value"]}
                        </Box>
                        <Divider
                            orientation="vertical"
                            flexItem
                            className={classes.dividerStyle}
                        />
                        <Box component="div" display="inline" m={2} mr={1}>
                            Participants:
                        </Box>
                        <Box
                            component="div"
                            display="inline"
                            fontWeight={"bold"}
                            m={2}
                            mr={6}
                        >
                            {"06"}
                        </Box>
                    </Grid>
                )}
                <Grid
                    container
                    alignItems="center"
                    className={classes.titlebar}
                >
                    <Box
                        component="div"
                        display="inline"
                        m={2}
                        mr={1}
                        fontWeight={"500"}
                    >
                        S no
                    </Box>
                    <Box
                        component="div"
                        display="inline"
                        m={2}
                        mr={1}
                        fontWeight={"500"}
                    >
                        {selectedTab === "Escalations"
                            ? SelectedCard === 1
                                ? "Teams and their " + selectedTab
                                : SelectedCard === 3 &&
                                  "Most Escalated Participants "
                            : selectedTab === "Appreciations" &&
                              SelectedCard === 1
                            ? "Teams and their " + selectedTab
                            : SelectedCard === 3 &&
                              "Most Appreciated Participants "}
                        {SelectedCard === 2 &&
                            "Participants and their " + selectedTab}
                    </Box>
                </Grid>
                <Box component="div">
                    <Divider
                        orientation="horizontal"
                        flexItem
                        className={classes.midDivider}
                    />
                </Box>
                <Grid className={`${classes.collapseScroll}`}>
                    <Grid item xs={12} direction="column">
                        <List
                            component="nav"
                            aria-labelledby="nested-list-subheader"
                        >
                            {Object.keys(getData).map((doc, index) => {
                                return (
                                    <CustomizedListItem
                                        index={index}
                                        doc={getData[doc]}
                                        title={doc}
                                    />
                                );
                            })}
                        </List>
                    </Grid>
                </Grid>
            </>
        );
    };

    const openSelectedPopup = (ele) => {
        setSelectedTeam(ele);
        openItemsPopup("Contributions", 2);
    };

    // console.log(
    //   '===selectedActionRowIdx===',
    //   selectedActionRowIdx,
    //   selectedActionRowData
    // );
    return (
        <>
            <Grid container spacing={2}>
                <Grid container item xs={10} direction="column">
                    <Typography className={classes.fontCss}>
                        {selectedTab} Overview
                    </Typography>
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
                            <Sort style={{ marginRight: "6px" }} /> Select Date
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
                            style={{ top: "21%" }}
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
            <Grid container spacing={2}>
                <Grid container item xs={4} direction="column">
                    <Paper className={classes.cards}>
                        <Grid
                            container
                            spacing={2}
                            justifyContent="space-between"
                            className={classes.cardFirstCss}
                        >
                            <Grid
                                container
                                item
                                xs={7}
                                direction="column"
                                m={1}
                                className={`${classes.textLeft} ${classes.fontSizeCss}`}
                            >
                                Recent {selectedTab}
                            </Grid>
                            <Grid
                                container
                                item
                                xs={4}
                                direction="column"
                                className={classes.marginCardFirstCss}
                            >
                                <Select
                                    autoFocus
                                    value={selectedValue}
                                    onChange={onChange}
                                    options={dropDownOptions}
                                    placeholder={"Select"}
                                    clearable={true}
                                    isSearchable={true}
                                />
                            </Grid>
                        </Grid>

                        <Box
                            component="div"
                            className={classes.heightFirstCard}
                        >
                            {result.map((ele, idx) => (
                                <>
                                    <Box sx={{ flexGrow: 1 }} m={2}>
                                        <Grid
                                            container
                                            spacing={2}
                                            justifyContent="space-between"
                                            className={classes.paddingStyle}
                                        >
                                            <Grid
                                                container
                                                item
                                                xs={8}
                                                direction="column"
                                                m={1}
                                                className={classes.textLeft}
                                            >
                                                {ele.name}
                                            </Grid>
                                            <Grid
                                                container
                                                item
                                                xs={2}
                                                direction="column"
                                            >
                                                {ele.value}
                                            </Grid>
                                        </Grid>
                                        <LinearProgress
                                            variant="determinate"
                                            value={"70"}
                                            ele={idx}
                                            sx={{
                                                height: 10,
                                                borderRadius: 5,
                                                [`&.${linearProgressClasses.colorPrimary}`]:
                                                    {
                                                        backgroundColor: (
                                                            theme
                                                        ) =>
                                                            theme.palette.grey[
                                                                theme.palette
                                                                    .mode ===
                                                                "light"
                                                                    ? 200
                                                                    : 800
                                                            ],
                                                    },
                                                [`& .${linearProgressClasses.bar}`]:
                                                    {
                                                        borderRadius: 5,
                                                        backgroundColor: (
                                                            theme
                                                        ) =>
                                                            theme.palette
                                                                .mode ===
                                                            "light"
                                                                ? colorPalette[
                                                                      idx
                                                                  ]
                                                                : "#308fe8",
                                                    },
                                            }}
                                        />
                                    </Box>
                                </>
                            ))}
                        </Box>
                        <Box
                            component="div"
                            display="flex"
                            color="#1665DF"
                            className={classes.viewallCss}
                            onClick={(e) => openItemsPopup("Recent", 1)}
                        >
                            {"View all"}
                            <ArrowForwardIosIcon
                                className={classes.fontcss}
                                sx={{ fontSize: 16 }}
                            />
                        </Box>
                    </Paper>
                </Grid>
                <Grid container item xs={4} direction="column">
                    <Paper className={classes.cards}>
                        <ReactECharts
                            option={Doughnut()}
                            style={{ height: "250px", width: "100%" }}
                            // onChartReady={onChartReadyCallback}
                        />
                        <Grid
                            container
                            spacing={2}
                            className={classes.borderCss}
                        >
                            {result.map((ele, idx) => (
                                <Grid
                                    container
                                    onClick={() => openSelectedPopup(ele)}
                                    className={classes.cursorCls}
                                >
                                    <Grid
                                        container
                                        item
                                        xs={1}
                                        direction="column"
                                        className={classes.dotBar}
                                    >
                                        <Box
                                            component="span"
                                            className={classes.dot}
                                            style={{
                                                background: colorPalette[idx],
                                            }}
                                        >
                                            {" "}
                                        </Box>
                                    </Grid>
                                    <Grid
                                        container
                                        item
                                        xs={7}
                                        direction="column"
                                        className={classes.textLeft}
                                    >
                                        {ele.name}
                                    </Grid>
                                    <Grid
                                        container
                                        item
                                        xs={3}
                                        direction="column"
                                        className={classes.paddingCss}
                                    >
                                        <Box component="span" display="flex">
                                            {ele.value}
                                            <Box
                                                component="span"
                                                className={classes.subtextCss}
                                            >
                                                (80%)
                                            </Box>
                                        </Box>
                                    </Grid>
                                </Grid>
                            ))}
                        </Grid>
                        <Box
                            component="div"
                            display="flex"
                            color="#1665DF"
                            className={classes.viewallCss}
                            onClick={(e) => openItemsPopup("Contributions", 2)}
                        >
                            {"View all"}
                            <ArrowForwardIosIcon
                                className={classes.fontcss}
                                sx={{ fontSize: 16 }}
                            />
                        </Box>
                    </Paper>
                </Grid>
                <Grid container item xs={4} direction="column">
                    <Paper className={classes.cards}>
                        <Grid
                            container
                            spacing={2}
                            justifyContent="space-between"
                            className={classes.cardFirstCss}
                        >
                            <Grid
                                container
                                item
                                xs={8}
                                direction="column"
                                m={1}
                                className={`${classes.textLeft} ${classes.fontSizeCss}`}
                            >
                                {`Most ${
                                    selectedTab === "Escalations"
                                        ? "Escalated"
                                        : "Appreciated"
                                } Participants`}
                            </Grid>
                        </Grid>
                        <Grid
                            container
                            spacing={2}
                            justifyContent="space-between"
                            className={`${classes.cardFirstCss} ${classes.heightLastCard}`}
                        >
                            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(
                                (item, idx) => (
                                    <>
                                        <Grid
                                            container
                                            spacing={2}
                                            className={classes.dividerLastCard}
                                        >
                                            <Grid
                                                container
                                                item
                                                xs={1}
                                                direction="column"
                                                className={`${classes.content} ${classes.widthFirstCard}`}
                                            >
                                                {idx
                                                    .toString()
                                                    .padStart(2, "0")}{" "}
                                                .
                                            </Grid>
                                            <Grid
                                                container
                                                item
                                                xs={8}
                                                direction="column"
                                                m={2}
                                                className={`${classes.content} ${classes.minWidthLastCard}`}
                                            >
                                                <Box
                                                    component="div"
                                                    display="flex"
                                                >
                                                    {
                                                        "Most Escalated Participants"
                                                    }
                                                </Box>
                                                <Box
                                                    component="div"
                                                    className={
                                                        classes.subtextCss
                                                    }
                                                >
                                                    {"Sense Maker"}
                                                </Box>
                                            </Grid>
                                            <Grid
                                                container
                                                item
                                                xs={1}
                                                direction="column"
                                                m={1}
                                                className={`${classes.circle} ${classes.WidthLastCard} `}
                                            >
                                                {"10"}
                                            </Grid>
                                        </Grid>
                                    </>
                                )
                            )}
                        </Grid>

                        <Box
                            component="div"
                            display="flex"
                            color="#1665DF"
                            className={classes.viewallCss}
                            onClick={(e) => openItemsPopup("Most ", 3)}
                        >
                            {"View all paticipants"}
                            <ArrowForwardIosIcon
                                className={classes.fontcss}
                                sx={{ fontSize: 16 }}
                            />
                        </Box>
                    </Paper>
                </Grid>
            </Grid>
            {(isOpenPopup || selectedActionRowIdx) && (
                <Modal
                    title={popupcontent.title}
                    content={popUPContent()}
                    open={true}
                    handleClose={handleModalClose}
                    // classesNamesDialog={momCls.modalpadding}
                    classesNamesTitle={momCls.modalTitleBar}
                    isContent={true}
                    width={"md"}
                    isCustomWidth={true}
                />
            )}
            {openPreview && (
                <PreviewFile
                    openPreview={true}
                    handlePreviewClose={handlePreviewClose}
                    fileName={fileName}
                    fileSize={fileSize}
                    src={fileSource}
                />
            )}
            ;
        </>
    );
};

export default EscalationsComp;
