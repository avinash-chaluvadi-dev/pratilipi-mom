import React, { useState, useEffect, useMemo } from "react";
import useStyles from "screens/FeedBackLoop/styles";
import {
    Box,
    Typography,
    Divider,
    Grid,
    Menu,
    MenuItem,
    Paper,
    ButtonGroup,
    Button,
    ListItem,
    List,
    ListItemText,
    TextField,
    Card,
    ClickAwayListener,
    IconButton,
    ListItemSecondaryAction,
} from "@material-ui/core";
import { OutlinedInput } from "@mui/material";
import Sort from "@mui/icons-material/Sort";
import globalSyles from "styles";
import ContentEditable from "react-contenteditable";
import { ContextMenu, ContextMenuTrigger } from "react-contextmenu";
import useSumaryStyles from "screens/FeedBackLoop/components/Summary/styles";
import useMoMStyles from "screens/FeedBackLoop/components/MoMView/useStyles";
import Modal from "components/Modal";
import PreviewFile from "screens/FeedBackLoop/components/Preview";
import DatePicker from "components/DatePicker";
import userIcon from "static/images/user.svg";
import videoIcon from "static/images/video.svg";
import imageViewIcon from "static/images/imageView.svg";
import ReactTooltip from "react-tooltip";
import openPopupIcon from "static/images/openPopupIcon.svg";
import lineConnectorIcon from "static/images/summarylineconnector.svg";
import DropDown from "components/DropDown";
import { AccountCircle, Edit } from "@mui/icons-material";
import Fade from "@mui/material/Fade";
import detailsViewStyles from "screens/FeedBackLoop/components/DetailedView/styles";
import { useDispatch, useSelector } from "react-redux";
import { updateFeedback } from "store/action/mom";
import {
    SummaryOptionsData,
    SummarySentimentData,
    MakerdataOptions,
    LabelDataOptions,
    SummaryEntitiesOptionsData,
    EntitiesDataOptions,
    SentimentDataOptions,
    ColorCheck,
    ButtonsDataRules,
} from "screens/FeedBackLoop/masterData";
import {
    prepareArr,
    markSelections,
    getStartEndStart,
    loadAssignedValues,
} from "screens/FeedBackLoop/commonLogic";
import TeamMember from "screens/FeedBackLoop/components/TeamMember";
import RightClickRules from "screens/FeedBackLoop/components/RightClickRules";
import PopupOutView from "screens/FeedBackLoop/components/PopupOutView";
import NoDataFound from "screens/FeedBackLoop/components/NoDataFound";

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

const Summary = (props) => {
    const classes = useStyles();
    const momCls = useMoMStyles();
    const detailCls = detailsViewStyles();
    const globalClasses = globalSyles();
    const summaryClasses = useSumaryStyles();

    const dispatch = useDispatch();
    // const getActions = useSelector((state) => state.addActions);
    const tabsValue = useSelector((state) => state.tabsReducer);
    const {
        momJson,
        meetingmetadata,
        momStore,
        isSaveChanges,
        redirection_mask_id,
    } = useSelector((state) => state.momReducer);

    let optionsdata = SummaryOptionsData();
    let sentimentdata = SummarySentimentData();
    let makerdata = MakerdataOptions();
    let Entitiesdata = SummaryEntitiesOptionsData();
    let labelOptions = LabelDataOptions();
    let sentimentOptions = SentimentDataOptions();
    let entitiesOptions = EntitiesDataOptions();
    const buttonsRules = ButtonsDataRules();

    let MoMStoreData = momStore?.concatenated_view || [];

    const [selectedBtn, setSelectedBtn] = useState(1);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [anchorEl, setAnchorEl] = useState(null);
    const open = Boolean(anchorEl);

    const [Transcript, setTranscript] = useState();
    const [selectedOptionVal, setSelectedOption] = useState("label");
    const [tooltipOpen, setTooltipOpen] = useState(false);
    const [mouseEnterVal, setMouseEnterVal] = useState(false);
    const [mouseOverCardVal, setMouseOverCard] = useState(null);
    const [isOpenPopup, setOpenPopup] = useState(null);
    const [selectedRowData, setSelectedRowData] = useState({});
    const [openPreview, setOpenPreview] = useState(false);
    const [fileName, setFileName] = useState("");
    const [fileSize, setFileSize] = useState("");
    const [fileSource, setFileSource] = useState("");
    const [isImageGallaryOpen, setImageGallaryOpen] = useState(false);
    const [optionsData, setOptionsData] = useState(optionsdata);
    const [SelectedItemVal, setSelectedItemVal] = useState(optionsdata[0]);
    const [usernameValue, setUserName] = useState("");
    const [openParticipant, setParticipantData] = useState(false);
    const [participantName, setParticipantName] = useState();
    const [isParticipantEdit, setParticipantEdit] = useState(false);
    const [anchorE2, setAnchorE2] = useState(null);
    const openParticipateMenu = Boolean(anchorE2);

    const [TeamName, setTeamName] = useState({});
    const [ScrumTeamName, setScrumTeamName] = useState(
        meetingmetadata?.meeting?.full_team_name
    );

    const [selectedIdx, setSelectedIdx] = useState(null);
    const [sortedValue, setSortedValue] = useState({
        start_time: null,
        speaker_id: null,
    });
    const [openContext, setOpen] = useState(false);
    const [contextMenu, setContextMenu] = useState(null);
    const [summaryTextObjects, setSummaryTextObjects] = useState({});
    const [startIndex, setStartIndex] = useState();
    const [endIndex, setEndIndex] = useState();
    const [selectedText, setSelectedText] = useState();
    const [textObjects, setTextObjects] = useState({});
    const [oldTextObjects, setOldTextObjects] = useState({});
    const [oldTeamName, setOldTeamName] = useState();
    const [ActionType, setActionType] = useState("");
    const [isOldData, setIsOldData] = useState("");
    const [oldSummaryTextObjects, setOldSummaryTextObjects] = useState({});
    const [previousJson, setPreviousJson] = useState(momJson);

    const updateJson = (arr) => {
        dispatch({
            type: "UPDATE_MOM_STORE",
            payload: { momStore: arr },
        });
    };

    const LoadDefaultData = () => {
        let textObj = {},
            summaryObj = {};
        MoMStoreData.forEach((ele, idx) => {
            textObj[idx] = ele.transcript;
            summaryObj[idx] = ele.summary;
        });
        setTextObjects(textObj);
        setOldTextObjects(textObj);
        setSummaryTextObjects(summaryObj);
        setOldSummaryTextObjects(summaryObj);
    };

    const handleToggle = (idx, type) => {
        setSelectedIdx(idx);
        setOpen(true);
    };

    const prepareData = () => {
        MoMStoreData = MoMStoreData.map((item, idx) => {
            return {
                ...item,
                label: prepareArr(item.label),
                sentiment: prepareArr(item.sentiment),
                marker: prepareArr(item.marker),
            };
        });
        momStore.concatenated_view = MoMStoreData;
        updateJson(momStore);
    };

    useMemo(() => {
        prepareData();
    }, []);

    useMemo(() => {
        // prepareArrayData();
        setPreviousJson(momJson);
        setSelectedItemVal(optionsdata[0]);

        MoMStoreData?.map((item, idx) => {
            let { words, type } = item.entities[0];
            let data = getStartEndStart(item.transcript, words, type);
            let final = markSelections(item.transcript, data);
            MoMStoreData[idx].transcript = final;
        });
        momStore.concatenated_view = MoMStoreData;
        updateJson(momStore);
        LoadDefaultData();
        // dispatch({ type: 'SWITCH_TABS', payload: { title: 2 } });
        dispatch({
            type: "STOP_LOADER",
            payload: { isLoader: false },
        });
    }, []);

    const isApplySaveChanges = (type) => {
        dispatch({
            type: "SAVE_CHANGES",
            payload: { isSaveChanges: true },
        });
    };
    const isCancelSaveChanges = (type) => {
        dispatch({
            type: "SAVE_CHANGES",
            payload: { isSaveChanges: false },
        });
    };

    const UpdateUserNamePopUp = (userName) => {
        setUserName({
            ...usernameValue,
            [selectedRowData?.idx]: userName,
        });
    };

    const updateUserName = (e, idx, userName, speaker_id, from) => {
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        if (from === "popup") {
            UpdateUserNamePopUp(userName);
        }
        setIsOldData({ speaker_id: momStore["map_username"][speaker_id] });
        momStore["map_username"][speaker_id] = usernameValue[idx];
        updateJson(momStore);
        hoverOn(idx);
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 2000);
        if (isOldData !== momStore["map_username"][speaker_id]) {
            isApplySaveChanges();
        }
        loadAssignToValues();
    };

    const loadAssignToValues = () => {
        let data = loadAssignedValues(momStore);
        updateJson(data);
    };

    useMemo(() => {
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 1000);
    }, [selectedOptionVal]);

    const filterLabelData = (items, optionsArr, type) => {
        items = items ? items : [];
        let final = optionsArr;
        let result = [];
        result = final.filter(function (item) {
            let values = item.value;

            return items.indexOf(values) > -1;
        });
        return result;
    };

    useEffect(() => {
        LoadDefaultData();
    }, [tabsValue]);

    const FieldsMapping = (value, item) => {
        switch (value) {
            case "Name":
                return { value: "Person Name", label: "Person Name" };
            default:
                return item;
        }
    };
    const handleListItemClick = (event, index, item) => {
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        setSelectedIndex(index);
        setSelectedItemVal(FieldsMapping(item.label, item));
        setTimeout(() => {
            dispatch({
                type: "START_LOADER",
                payload: { isLoader: false },
            });
        }, 2000);
    };

    const handleClick = (event) => {
        setAnchorEl(event.currentTarget);
    };
    const handleClose = () => {
        setAnchorEl(null);
    };

    const hoverOn = (idx) => {
        setMouseOverCard(idx);
        setMouseEnterVal(true);
    };

    const hoverOff = () => {
        setMouseEnterVal(false);
    };

    const handlePreviewOpen = (rowData) => {
        setOpenPreview(true);
        setFileName(momStore.file_name);
        setFileSize(momStore.size);
        setFileSource(rowData.video_path);
    };

    const handlePreviewClose = (e) => {
        setOpenPreview(false);
    };

    const imageGallaryOpen = (item, idx) => {
        setImageGallaryOpen(true);
        setSelectedRowData({ item, idx });
        setOpenPopup(true);
    };

    const handleTooltip = (bool) => {
        setTooltipOpen(bool);
    };

    const handleChangeDropDownLabel = async (selectedOption, idx) => {
        // setSelectedOptionLabel(selectedOption);
        // selectedOption.value = selectedOption.value.slice(0, -1);
        if (MoMStoreData[idx].label.indexOf(selectedOption.value) === -1) {
            MoMStoreData[idx].bkp_label = {
                ...MoMStoreData[idx].bkp_label,
                [selectedOption.value]: 0.01,
            };
            MoMStoreData[idx].label.push(selectedOption.value);
            momStore.concatenated_view = MoMStoreData;
            updateJson(momStore);
            hoverOff();
        } else {
            MoMStoreData[idx].label = MoMStoreData[idx].label.filter(
                (item) => item !== selectedOption.value
            );
            delete MoMStoreData[idx].bkp_label[selectedOption.value];
            momStore.concatenated_view = MoMStoreData;
            updateJson(momStore);
            hoverOff();
        }
        compareData();
    };

    const handleDateChange = (data, idx) => {
        MoMStoreData[idx].date = data;
        momStore.concatenated_view = MoMStoreData;
        // setTranscript(MoMStoreData);
        updateJson(momStore);
        compareData();
    };

    const handleChangeDropDownEntity = (selectedOption, idx) => {
        // setSelectedOptionEntity(selectedOption);
        if (selectedText === "") {
            alert("please select the word");
            return;
        }
        if (
            MoMStoreData[idx]?.entities[0]?.type.indexOf(
                selectedOption.value
            ) === -1
        ) {
            if (selectedOption.value === "Name") {
                selectedOption.value = "Person Name";
            }
            MoMStoreData[idx].entities[0].type.push(selectedOption.value);
            MoMStoreData[idx].entities[0].words.push(selectedText);
        } else {
            MoMStoreData[idx].entities[0].type = MoMStoreData[
                idx
            ]?.entities[0]?.type.filter(
                (item) => item !== selectedOption.value
            );
            // MoMStoreData[idx].entities[0].words = MoMStoreData[
            //   idx
            // ]?.entities[0]?.words.filter((item) => item !== selectedOption.value);
        }
        MoMStoreData?.map((item, idx) => {
            let { words, type } = item.entities[0];
            let data = getStartEndStart(item.transcript, words, type);
            let final = markSelections(item.transcript, data);
            MoMStoreData[idx].transcript = final;
        });
        momStore.concatenated_view = MoMStoreData;
        updateJson(momStore);
        hoverOff();
        compareData();
        LoadDefaultData();
        setSelectedText("");
    };

    const handleChangeDropDownSentiments = async (selectedOption, idx) => {
        // setSelectedOptionSentiments(selectedOption);
        MoMStoreData[idx].sentiment = MoMStoreData[idx].sentiment
            ? MoMStoreData[idx].sentiment
            : [];
        if (MoMStoreData[idx].sentiment.indexOf(selectedOption.value) === -1) {
            MoMStoreData[idx].sentiment = MoMStoreData[idx].sentiment.filter(
                (s) => typeof s !== "string"
            );
            MoMStoreData[idx].bkp_sentiment = {
                [selectedOption.value]: 0.01,
            };
            MoMStoreData[idx].sentiment.push(selectedOption.value);
            momStore.concatenated_view = MoMStoreData;
            updateJson(momStore);
            hoverOff();
        } else {
            MoMStoreData[idx].sentiment = MoMStoreData[idx].sentiment.filter(
                (item) => item !== selectedOption.value
            );
            delete MoMStoreData[idx].bkp_sentiment[selectedOption.value];
            momStore.concatenated_view = MoMStoreData;
            updateJson(momStore);
            hoverOff();
        }
        compareData();
    };

    const openModal = (item, idx) => {
        setSelectedRowData({ item, idx });
        setOpenPopup(true);
    };

    const handleModalClose = () => {
        setOpenPopup(false);
        if (isImageGallaryOpen) {
            setImageGallaryOpen(false);
        }
    };

    //---participant---
    const OpenParticipateMenu = (event) => {
        setAnchorE2(event.currentTarget);
        setParticipantEdit(true);
    };

    const CloseParticipateMenu = () => {
        setAnchorE2(null);
    };

    const addNewParticipatient = (e, type) => {
        setParticipantData(true);
        setAnchorE2(null);
        setActionType("add");
    };

    const closeParticipant = (event, type) => {
        setParticipantData(false);
    };

    const onchangeParticipant = (e, type) => {
        setParticipantName(e.target.value);
    };

    const CancelParticipant = (e, type) => {
        setParticipantName("");
        setParticipantData(false);
    };

    const compareData = (idx) => {
        if (idx === undefined) {
            isApplySaveChanges();
        } else if (
            JSON.stringify(oldTextObjects[idx]) !==
                JSON.stringify(textObjects[idx]) ||
            JSON.stringify(oldSummaryTextObjects[idx]) !==
                JSON.stringify(summaryTextObjects[idx])
        ) {
            isApplySaveChanges();
        }
    };

    const AddParticipantBtn = (e, type) => {
        if (ActionType === "edit") {
            momStore["map_username"][oldTeamName[type].speaker_id] =
                participantName;
        }
        if (ActionType === "add") {
            let arr = momStore["ExternalParticipants"]
                ? momStore["ExternalParticipants"]
                : [];
            momStore["ExternalParticipants"] = [
                ...arr,
                {
                    value: participantName,
                    label: participantName,
                    speaker_id: "",
                },
            ];
            updateJson(momStore);
        }
        loadAssignToValues();
        setParticipantName("");
        setParticipantData(false);
    };

    const EditParticipantData = (e, team) => {
        setParticipantData(true);
        setParticipantEdit(false);
        setParticipantName(team.value);
        setOldTeamName({ ...TeamName, Participants: team });
        setAnchorE2(null);
        setActionType("edit");
        compareData();
    };

    const handelLabelSelection = async () => {
        await setOptionsData(optionsdata);
        await setSelectedOption("label");
        await setSelectedIndex(0);
        await setSelectedItemVal(optionsdata[0]);
    };

    const handelEntitySelection = () => {
        setSelectedOption("entities");
        setOptionsData(Entitiesdata);
        setSelectedIndex(0);
        setSelectedItemVal(Entitiesdata[0]);
    };

    const handelMakerSelection = () => {
        setOptionsData(makerdata);
        setSelectedOption("marker");
        setSelectedIndex(0);
        setSelectedItemVal(makerdata[0]);
    };

    const handelSentimentSelection = async () => {
        await setOptionsData(sentimentdata);
        await setSelectedOption("sentiment");
        setSelectedIndex(0);
        setSelectedItemVal(sentimentdata[0]);
    };

    const sortByName = (type, sortedValue) => {
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        let arr = MoMStoreData.sort(function (a, b) {
            if (
                momStore["map_username"][a[type]] <
                    momStore["map_username"][b[type]] &&
                !sortedValue[type]
            ) {
                setSortedValue({ ...sortedValue, [type]: true });
                return -1;
            }
            if (
                momStore["map_username"][a[type]] >
                    momStore["map_username"][b[type]] &&
                sortedValue[type]
            ) {
                setSortedValue({ ...sortedValue, [type]: false });
                return 1;
            }
            return 0;
        });
        handleClose();
        // setTranscript(arr);
        momStore.concatenated_view = arr;
        updateJson(momStore);
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 1000);
    };

    const sortByTime = (type, sortedValue) => {
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        const getNumberFromTime = (time) => +time.replace(/:/g, "");
        let arr;
        if (!sortedValue[type]) {
            arr = MoMStoreData.sort(
                (a, b) =>
                    getNumberFromTime(a.start_time) -
                    getNumberFromTime(b.start_time)
            );
            setSortedValue({ ...sortedValue, [type]: true });
        } else {
            arr = MoMStoreData.sort(
                (a, b) =>
                    getNumberFromTime(b.start_time) -
                    getNumberFromTime(a.start_time)
            );
            setSortedValue({ ...sortedValue, [type]: false });
        }
        handleClose();
        // setTranscript(arr);
        momStore.concatenated_view = arr;
        updateJson(momStore);
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 500);
    };

    const handleUserName = (value, item, idx) => {
        if (item?.startsWith("User 0") && item?.split("User 0").length > 1) {
            return value || item;
        }
        return value || "User 0" + item;
    };

    const handleChangeTemp = (event) => {
        // text.current = evt.target.value;
        let final = { ...MoMStoreData };
        final[selectedIdx].transcript = event.target.value;
        setTranscript(MoMStoreData);
        setTextObjects({ ...textObjects, [selectedIdx]: event.target.value });
        isApplySaveChanges();
    };

    const filterEntitiesData = (items, optionsArr) => {
        items = items ? items[0]?.type : [];
        let final = optionsArr;
        let result = [];
        result = final.filter(function (item) {
            let dataVal = item.value;
            dataVal = dataVal === "Name" ? "Person Name" : dataVal;
            return items.indexOf(dataVal) > -1;
        });
        return result;
    };

    const handelActionsData = () => {
        let data = MoMStoreData.map((item) => {
            if (
                item[selectedOptionVal] &&
                item[selectedOptionVal].indexOf(SelectedItemVal.label) !== -1
            ) {
                return item.chunk_id;
            }
        });
        dispatch({
            type: "SWITCH_TABS",
            payload: {
                title: 2,
                isUserSelection: true,
                isHilightCard: true,
                userName: SelectedItemVal.value,
                highlitedGroup: data,
                from: "summary",
            },
        });
    };

    const labelsCount = (value) => {
        let fieldName = selectedOptionVal;
        // fieldName = fieldName.slice(0, -1);
        if (selectedOptionVal === "entities") {
            let TranscriptArr =
                MoMStoreData &&
                MoMStoreData.filter((item, idx) => {
                    return (
                        item[fieldName] &&
                        item[fieldName][0]?.type?.indexOf(
                            value === "Name" ? "Person Name" : value
                        ) !== -1
                    );
                }).length;
            return TranscriptArr;
        }
        let TranscriptArr =
            MoMStoreData &&
            MoMStoreData.filter((item, idx) => {
                return (
                    item[fieldName] && item[fieldName]?.indexOf(value) !== -1
                );
            }).length;
        return TranscriptArr;
    };

    const onFocus = (idx) => {
        setSelectedIdx(idx);
    };

    const onInputchange = (event) => {
        let final = { ...MoMStoreData };
        final[selectedIdx].transcript = event.target.value;
        setTranscript(MoMStoreData);
        setTextObjects({ ...textObjects, [selectedIdx]: event.target.value });
    };

    const onSummaryInputchange = (event, idx) => {
        let final = { ...MoMStoreData };
        final[selectedIdx].summary = event.target.value;
        setTranscript(MoMStoreData);
        setSummaryTextObjects({
            ...summaryTextObjects,
            [idx]: event.target.value,
        });
    };

    const saveChangesAPICall = () => {
        isCancelSaveChanges();
        MoMStoreData = MoMStoreData.map((item) => {
            return {
                ...item,
                speaker_id: momStore["map_username"][item.speaker_id],
                label: item.bkp_label,
                sentiment: item.bkp_sentiment,
                marker: item.bkp_marker,
            };
        });
        momStore.concatenated_view = MoMStoreData;
        dispatch(updateFeedback(redirection_mask_id, momStore));
        prepareData();
    };

    const handleCloseMenu = () => {
        setOpen(false);
        setContextMenu(null);
    };
    const handleSelectedText = (e) => {
        let data = window.getSelection().getRangeAt(0);
        let startIdx = data.startOffset;
        let endIdx = data.endOffset;
        let selectedTextArea = document.getSelection().toString();
        // const textArea = e.target.value.substring(startIdx, endIdx);
        setStartIndex(startIdx);
        setEndIndex(endIdx);
        setSelectedText(selectedTextArea);
    };

    const mainChildComponent = (idx, item) => {
        return (
            <>
                <Grid item xs={6} className={classes.actionCardCss}>
                    <Paper
                        className={`${summaryClasses.paperCommonCss} ${
                            mouseOverCardVal === idx && mouseEnterVal
                                ? summaryClasses.paperHoverStyle
                                : ""
                        }`}
                        onMouseEnter={() => hoverOn(idx)}
                        onMouseLeave={() => hoverOff(idx)}
                    >
                        <Grid
                            container
                            className={classes.rootGrid}
                            spacing={2}
                        >
                            <Grid
                                item
                                xs={12}
                                p={0}
                                className={classes.profileBarRight}
                            >
                                <img
                                    src={userIcon}
                                    height="20px"
                                    width="20px"
                                    alt=""
                                />
                                <TextField
                                    type="text"
                                    margin="normal"
                                    variant="outlined"
                                    size={"small"}
                                    className={summaryClasses.userText}
                                    placeholder="User name"
                                    value={
                                        usernameValue[idx] ||
                                        (momStore["map_username"] &&
                                            momStore["map_username"][
                                                item?.speaker_id
                                            ])
                                    }
                                    onChange={(e) =>
                                        setUserName(
                                            {
                                                ...usernameValue,
                                                [idx]: e.target.value,
                                            }
                                            // idx
                                        )
                                    }
                                    onBlur={(e) =>
                                        updateUserName(
                                            e,
                                            idx,
                                            momStore["map_username"][
                                                item?.speaker_id
                                            ],
                                            item?.speaker_id
                                        )
                                    }
                                    InputProps={{
                                        style: { width: "80px" },
                                    }}
                                />
                                <Typography className={classes.timer}>
                                    {item?.start_time
                                        ? item?.start_time
                                        : "00:00:00"}
                                    {" - "}
                                    {item?.end_time
                                        ? item?.end_time
                                        : "00:00:00"}
                                </Typography>
                                <ReactTooltip delayShow={500} />
                                {/* {mouseOverCardVal === idx && mouseEnterVal && ( */}
                                <img
                                    src={imageViewIcon}
                                    height="20px"
                                    width="20px"
                                    className={classes.videoIcon}
                                    data-tip="Show Frames"
                                    onClick={() => imageGallaryOpen(item, idx)}
                                    alt=""
                                />
                                {/* )} */}
                                <ReactTooltip delayShow={500} />
                                {/* {mouseOverCardVal === idx && mouseEnterVal && ( */}
                                <img
                                    src={videoIcon}
                                    height="20px"
                                    width="20px"
                                    className={classes.videoIcon}
                                    data-tip="Play Files"
                                    onClick={() => handlePreviewOpen(item)}
                                    alt=""
                                />
                                {/* )} */}
                                <DropDown
                                    // value={selectedOptionEntity && selectedOptionEntity[idx]}
                                    value={
                                        filterEntitiesData(
                                            item?.entities,
                                            entitiesOptions
                                        ).length > 1
                                            ? [
                                                  {
                                                      label:
                                                          "entities" +
                                                          `(${
                                                              filterEntitiesData(
                                                                  item?.entities,
                                                                  entitiesOptions
                                                              ).length
                                                          })`,
                                                      value:
                                                          "entities" +
                                                          `(${
                                                              filterEntitiesData(
                                                                  item?.entities,
                                                                  entitiesOptions
                                                              ).length
                                                          })`,
                                                  },
                                                  ...filterEntitiesData(
                                                      item?.entities,
                                                      entitiesOptions
                                                  ),
                                              ]
                                            : filterEntitiesData(
                                                  item?.entities,
                                                  entitiesOptions
                                              )
                                    }
                                    onChange={(e) =>
                                        handleChangeDropDownEntity(e, idx)
                                    }
                                    options={entitiesOptions}
                                    type={"Entity"}
                                    placeholder={"Entities"}
                                    isNormal={false}
                                    isSearchable={false}
                                    isMulti={true}
                                    isScroll={true}
                                />
                            </Grid>
                            {/* <TextInput value={item.transcript} /> */}
                            <div
                                onClick={() => handleToggle(idx, "transcript")}
                            >
                                <ContextMenuTrigger id="folder-context-menu">
                                    <ContentEditable
                                        html={textObjects[idx]}
                                        // onBlur={handleBlur}
                                        onChange={handleChangeTemp}
                                        disabled={false}
                                        value={textObjects[idx]}
                                        onSelect={(e) =>
                                            handleSelectedText(e, idx)
                                        }
                                        onFocus={() => onFocus(idx)}
                                        onBlur={() => compareData(idx)}
                                        style={{
                                            height: "46px",
                                            overflowY: "scroll",
                                            width: "495px",
                                            textAlign: "left",
                                            color: "rgba(0, 0, 0, 0.95)",
                                            fontFamily: "sans-serif",
                                            fontSize: "13.2px",
                                            lineBreak: "initial",
                                            lineHeight: "16px",
                                            letterSpacing: "0.3px",
                                        }}
                                    />
                                </ContextMenuTrigger>
                            </div>
                        </Grid>
                        <Box component="div" display="flex">
                            <img
                                src={lineConnectorIcon}
                                height="10px"
                                width="10px"
                                className={summaryClasses.connectorSummary}
                                alt=""
                            />
                        </Box>
                    </Paper>
                </Grid>

                <Grid item xs={6} className={classes.actionCardCss}>
                    <Paper
                        className={`${summaryClasses.paperCommonCssSecond}  ${
                            mouseOverCardVal === idx && mouseEnterVal
                                ? summaryClasses.paperHoverStyle
                                : ""
                        }`}
                        onMouseEnter={() => hoverOn(idx)}
                        onMouseLeave={() => hoverOff(idx)}
                    >
                        <Grid
                            container
                            className={classes.rootGrid}
                            spacing={2}
                        >
                            <Grid
                                item
                                xs={12}
                                p={0}
                                className={classes.profileBarRight}
                            >
                                <Typography className={classes.timerRight}>
                                    {item?.end_time || "00:00:00"}
                                </Typography>

                                {
                                    //  {/* <Tooltip title="Change Label" placement="top" open={tooltipOpen} arrow> */}
                                    <DropDown
                                        // value={selectedOptionLabel}
                                        value={
                                            filterLabelData(
                                                item?.label,
                                                labelOptions
                                            ).length > 1
                                                ? [
                                                      {
                                                          label:
                                                              "Labels" +
                                                              `(${
                                                                  filterLabelData(
                                                                      item?.label,
                                                                      labelOptions
                                                                  ).length
                                                              })`,
                                                          value:
                                                              "Labels" +
                                                              `(${
                                                                  filterLabelData(
                                                                      item?.label,
                                                                      labelOptions
                                                                  ).length
                                                              })`,
                                                      },
                                                      ...filterLabelData(
                                                          item?.label,
                                                          labelOptions
                                                      ),
                                                  ]
                                                : filterLabelData(
                                                      item?.label,
                                                      labelOptions
                                                  )
                                        }
                                        onChange={(e) =>
                                            handleChangeDropDownLabel(e, idx)
                                        }
                                        options={labelOptions}
                                        type={"Label"}
                                        isNormal={false}
                                        isSearchable={false}
                                        placeholder={"Labels"}
                                        from={"summary"}
                                    />
                                    // )
                                }

                                <TeamMember
                                    type="Organizer"
                                    idx={idx}
                                    value={item?.assign_to}
                                    compareData={compareData}
                                    loadAssignToValues={loadAssignToValues}
                                />

                                <DatePicker
                                    placeholder="Select date"
                                    value={item?.date || null}
                                    handleDateChange={(e) =>
                                        handleDateChange(e, idx)
                                    }
                                    idx={idx}
                                />

                                <DropDown
                                    // value={selectedOptionSentiments}
                                    value={
                                        filterLabelData(
                                            item?.sentiment,
                                            sentimentOptions,
                                            "sentiment"
                                        ).length > 1
                                            ? [
                                                  {
                                                      label:
                                                          "sentiments" +
                                                          `(${
                                                              filterLabelData(
                                                                  item?.sentiment,
                                                                  sentimentOptions,
                                                                  "sentiment"
                                                              ).length
                                                          })`,
                                                      value:
                                                          "sentiments" +
                                                          `(${
                                                              filterLabelData(
                                                                  item?.sentiment,
                                                                  sentimentOptions,
                                                                  "sentiment"
                                                              ).length
                                                          })`,
                                                  },
                                                  ...filterLabelData(
                                                      item?.sentiment,
                                                      sentimentOptions,
                                                      "sentiment"
                                                  ),
                                              ]
                                            : filterLabelData(
                                                  item?.sentiment,
                                                  sentimentOptions,
                                                  "sentiment"
                                              )
                                    }
                                    onChange={(e) =>
                                        handleChangeDropDownSentiments(e, idx)
                                    }
                                    options={sentimentOptions}
                                    type={"sentimentIcon"}
                                    isNormal={false}
                                    isSearchable={false}
                                    placeholder={"sentiments"}
                                    from={"summary"}
                                />
                                {mouseOverCardVal === idx && mouseEnterVal && (
                                    <Box
                                        component="span"
                                        className={classes.openPopup}
                                    >
                                        <ReactTooltip delayShow={500} />
                                        <img
                                            src={openPopupIcon}
                                            height="16px"
                                            width="16px"
                                            className={`${classes.videoIcon}`}
                                            data-tip="Open Popup"
                                            onClick={() => openModal(item, idx)}
                                            alt=""
                                        />
                                    </Box>
                                )}
                            </Grid>
                            {/* <TextInput value={item.summary} /> */}
                            <TextField
                                // disabled
                                // autoFocus
                                type="text"
                                multiline
                                rows={2}
                                margin="normal"
                                variant="outlined"
                                size={"small"}
                                fullWidth
                                placeholder="Enter Text Here"
                                value={summaryTextObjects[idx]}
                                className={detailCls.userText}
                                onChange={(e) => {
                                    onSummaryInputchange(e, idx);
                                }}
                                onFocus={() => onFocus(idx)}
                                onBlur={() => compareData(idx)}
                                InputProps={{
                                    classes: {
                                        input: classes.thaiTextFieldInputProps,
                                    },
                                    // style: { width: '550px' },
                                }}
                            />
                        </Grid>
                    </Paper>
                </Grid>
            </>
        );
    };

    const updateDataEvent = (textObjectsVal) => {
        setTextObjects(textObjectsVal);
        handleCloseMenu();
        setSelectedText("");
        setStartIndex("");
        setEndIndex("");
        handleCloseMenu();
        isApplySaveChanges();
    };

    return (
        <Grid container alignItems="center">
            <Box
                component="div"
                display="inline"
                // fontWeight={'bold'}
                m={2}
                mr={1}
            >
                Team name:
            </Box>
            <Box
                component="div"
                display="inline"
                fontWeight={"bold"}
                m={2}
                mr={6}
            >
                {ScrumTeamName}
            </Box>
            <Divider
                orientation="vertical"
                flexItem
                className={classes.dividerStyle}
            />

            <Button
                id="fade-button"
                aria-controls="fade-menu"
                aria-haspopup="true"
                aria-expanded={openParticipateMenu ? "true" : undefined}
                onClick={OpenParticipateMenu}
                className={summaryClasses.participants}
            >
                Participants:{" "}
                <b>{("0" + momStore?.AssignTo?.length)?.slice(-2)}</b>
            </Button>
            <Menu
                id="fade-menu"
                MenuListProps={{
                    "aria-labelledby": "fade-button",
                }}
                anchorEl={anchorE2}
                open={openParticipateMenu}
                onClose={CloseParticipateMenu}
                TransitionComponent={Fade}
                style={{ top: "40px" }}
            >
                <MenuItem
                    value=""
                    onClick={(event) => {
                        addNewParticipatient(event);
                    }}
                    className={momCls.addTeam}
                >
                    + Add Participatent
                </MenuItem>
                {momStore?.AssignTo?.map((team) => (
                    <MenuItem key={Math.random()} value={team.value}>
                        <AccountCircle className={momCls.userIcon} />
                        {team.value}
                        {isParticipantEdit ? (
                            <ListItemSecondaryAction>
                                <IconButton
                                    edge="end"
                                    className={momCls.editBlock}
                                    onClick={(e) =>
                                        EditParticipantData(e, team)
                                    }
                                >
                                    <Edit className={momCls.editIcon} />
                                </IconButton>
                            </ListItemSecondaryAction>
                        ) : null}
                    </MenuItem>
                ))}
            </Menu>
            {openParticipant && (
                <ClickAwayListener onClickAway={(e) => closeParticipant(e)}>
                    <Card
                        className={summaryClasses.addTeamPopUp}
                        style={{ top: "400px" }}
                    >
                        <Box padding="10px">
                            <TextField
                                fullWidth
                                variant="outlined"
                                size="small"
                                value={participantName}
                                // autoFocus={true}
                                onChange={(event) => {
                                    onchangeParticipant(event);
                                }}
                                placeholder="Enter teamname"
                                input={<OutlinedInput />}
                                className={momCls.projectTextcls}
                            />
                        </Box>
                        <Box alignItems="right" marginLeft="140px">
                            <Button
                                variant="text"
                                onClick={(e) => CancelParticipant(e)}
                            >
                                cancel
                            </Button>
                            <Button
                                variant="contained"
                                color="primary"
                                component="label"
                                style={{
                                    textTransform: "none",
                                    fontSize: "16px",
                                }}
                                size="small"
                                onClick={(e) =>
                                    AddParticipantBtn(e, "Participants")
                                }
                            >
                                Submit
                            </Button>
                        </Box>
                    </Card>
                </ClickAwayListener>
            )}
            <Grid
                container
                item
                xs={8}
                direction="column"
                className={detailCls.midblockright}
            >
                <Button
                    variant="contained"
                    color="primary"
                    onClick={saveChangesAPICall}
                    className={classes.button}
                    style={{ textTransform: "none", margin: "0 20px 0 10px" }}
                    disabled={!isSaveChanges}
                >
                    Save changes
                </Button>
            </Grid>
            <Box
                component="div"
                display="block"
                width={"100%"}
                b={3}
                className={classes.horizontalDividerDiv}
            >
                <Divider
                    variant="middle"
                    className={classes.horizontalDivider}
                    mb={30}
                />
            </Box>
            <Box
                display="block"
                component="div"
                width={"100%"}
                justifyContent="space-between"
                mr={1}
                pt={0}
                p={2}
            >
                <ButtonGroup
                    aria-label="outlined primary button group"
                    disableElevation
                    variant="contained"
                    color="#0000"
                    style={{ border: "1px solid gray" }}
                >
                    <Button
                        className={`${globalClasses.bold} ${classes.transformvalue}`}
                        style={
                            selectedBtn === 1
                                ? {
                                      background: "#1563DB",
                                      color: "#fff",
                                      padding: "8px 45px",
                                      margin: "-1px",
                                  }
                                : { background: "white", padding: "8px 45px" }
                        }
                        onClick={() => {
                            setSelectedBtn(1);
                            handelLabelSelection("label");
                        }}
                    >
                        Labels
                    </Button>
                    <Button
                        className={`${globalClasses.bold} ${classes.transformvalue}`}
                        style={
                            selectedBtn === 1
                                ? {
                                      background: "#1563DB",
                                      color: "#fff",
                                      padding: "8px 45px",
                                      margin: "-1px",
                                  }
                                : { background: "white", padding: "8px 45px" }
                        }
                        onClick={() => {
                            setSelectedBtn(1);
                            handelEntitySelection("Entity");
                        }}
                    >
                        Entities
                    </Button>
                    <Button
                        className={`${globalClasses.bold} ${classes.transformvalue}`}
                        style={
                            selectedBtn === 3
                                ? {
                                      background: "#1563DB",
                                      color: "#fff",
                                      padding: "8px 45px",
                                      margin: "-1px",
                                  }
                                : { background: "white", padding: "8px 45px" }
                        }
                        onClick={() => {
                            setSelectedBtn(3);
                            handelMakerSelection("Maker");
                        }}
                    >
                        Markers
                    </Button>
                    <Button
                        className={`${globalClasses.bold} ${classes.transformvalue}`}
                        style={
                            selectedBtn === 4
                                ? {
                                      background: "#1563DB",
                                      color: "#fff",
                                      padding: "8px 45px",
                                      margin: "-1px",
                                  }
                                : { background: "white", padding: "8px 45px" }
                        }
                        onClick={() => {
                            setSelectedBtn(4);
                            handelSentimentSelection("Sentiment");
                        }}
                    >
                        Sentiments
                    </Button>
                </ButtonGroup>
            </Box>

            <Grid container spacing={2} className={summaryClasses.mainGrid}>
                <Grid container item xs={2} direction="column">
                    <List
                        component="nav"
                        aria-label="secondary mailbox folder"
                        className={summaryClasses.listroot}
                    >
                        {optionsData.map((item, idx) => (
                            <ListItem
                                button
                                selected={selectedIndex === idx}
                                onClick={(event) =>
                                    handleListItemClick(event, idx, item)
                                }
                                className={
                                    selectedIndex === idx
                                        ? summaryClasses.listItemSelected
                                        : summaryClasses.listItems
                                }
                                style={{ color: ColorCheck(item["value"]) }}
                            >
                                <ListItemText
                                    primary={
                                        item.value +
                                        " (" +
                                        labelsCount(item.label) +
                                        ")"
                                    }
                                />
                            </ListItem>
                        ))}
                    </List>
                </Grid>
                <Grid container item xs={10} direction="column">
                    <Box component="div" className={summaryClasses.sortcls}>
                        <Button
                            id="basic-button"
                            aria-controls="basic-menu"
                            aria-haspopup="true"
                            aria-expanded={open ? "true" : undefined}
                            style={
                                open
                                    ? { background: "lightgray" }
                                    : { background: "" }
                            }
                            onClick={handleClick}
                            className={`${classes.transformvalue}`}
                        >
                            <Sort style={{ marginRight: "10px" }} />
                            Sort
                        </Button>
                        <Menu
                            id="basic-menu"
                            anchorEl={anchorEl}
                            open={open}
                            onClose={handleClose}
                            MenuListProps={{
                                "aria-labelledby": "basic-button",
                            }}
                            style={{ top: "45px" }}
                        >
                            <MenuItem
                                onClick={() =>
                                    sortByName("speaker_id", sortedValue)
                                }
                            >
                                Sort by name
                            </MenuItem>
                            <MenuItem
                                onClick={() =>
                                    sortByTime("start_time", sortedValue)
                                }
                            >
                                Sort by time
                            </MenuItem>
                        </Menu>
                        <Typography
                            style={{ fontWeight: "600", cursor: "pointer" }}
                            onClick={handelActionsData}
                        >{`Show all ${SelectedItemVal.value} in detailed view`}</Typography>
                    </Box>

                    <Box
                        display="block"
                        component="div"
                        width={"100%"}
                        mt={1}
                        mb={3}
                    >
                        <Divider mt={10} mb={10} />
                    </Box>
                    <Grid container spacing={2}>
                        <Grid
                            container
                            item
                            xs={6}
                            direction="column"
                            className={summaryClasses.headercls}
                        >
                            <Typography className={summaryClasses.boldStyle}>
                                Transcript
                            </Typography>
                        </Grid>
                        <Grid
                            container
                            item
                            xs={6}
                            direction="column"
                            className={summaryClasses.headercls}
                        >
                            <Typography className={summaryClasses.boldStyle}>
                                Transcript Summary
                            </Typography>
                        </Grid>
                    </Grid>
                    <Box
                        display="block"
                        component="div"
                        width={"100%"}
                        mt={1}
                        mb={0}
                    >
                        <Divider mt={10} mb={10} />
                    </Box>
                    <Grid
                        container
                        spacing={1}
                        width={"100%"}
                        className={summaryClasses.summarydetails}
                    >
                        {MoMStoreData.length > 0 &&
                            MoMStoreData?.map((item, idx) => {
                                return item[selectedOptionVal] &&
                                    item[selectedOptionVal].indexOf(
                                        SelectedItemVal.label
                                    ) !== -1
                                    ? mainChildComponent(idx, item)
                                    : selectedOptionVal === "entities" &&
                                      item[selectedOptionVal] &&
                                      item[selectedOptionVal][0]?.type?.indexOf(
                                          // SelectedItemVal.label === 'Person Name'
                                          //   ? 'Name'
                                          SelectedItemVal.label
                                      ) !== -1
                                    ? mainChildComponent(idx, item)
                                    : labelsCount(SelectedItemVal.label) ===
                                          0 &&
                                      idx === 0 && (
                                          <NoDataFound
                                              margin={"11rem 0rem 0 28rem"}
                                              size={12}
                                          />
                                      );
                            })}
                        {MoMStoreData.length === 0 && (
                            <NoDataFound
                                margin={"11rem 0rem 0 28rem"}
                                size={12}
                            />
                        )}
                    </Grid>
                </Grid>
            </Grid>
            {isOpenPopup && (
                <Modal
                    title={"Pop Out View"}
                    content={
                        <PopupOutView
                            usernameValueText={usernameValue}
                            selectedRowData={selectedRowData}
                            loadAssignToValues={loadAssignToValues}
                            isApplySaveChanges={isApplySaveChanges}
                            hoverOn={hoverOn}
                            handleChangeTemp={handleChangeTemp}
                            handleSelectedText={handleSelectedText}
                            onFocus={onFocus}
                            compareData={compareData}
                            tooltipOpen={tooltipOpen}
                            filterLabelData={filterLabelData}
                            labelOptions={labelOptions}
                            handleChangeDropDownLabel={
                                handleChangeDropDownLabel
                            }
                            handleDateChange={handleDateChange}
                            sentimentOptions={sentimentOptions}
                            handleChangeDropDownSentiments={
                                handleChangeDropDownSentiments
                            }
                            summaryTextObjects={summaryTextObjects}
                            textObjects={textObjects}
                            onSummaryInputchange={onSummaryInputchange}
                            isImageGallaryOpen={isImageGallaryOpen}
                            TeamMember={TeamMember}
                            updateUserName={updateUserName}
                        />
                    }
                    actions={""}
                    width={"md"}
                    open={true}
                    handleClose={handleModalClose}
                    classesNamesDialog={classes.modalWidtHeight}
                    classesNamesTitle={classes.modalTitleBar}
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
            {openContext && (
                <ClickAwayListener onClickAway={handleCloseMenu}>
                    <ContextMenu
                        id="folder-context-menu"
                        className={detailCls.contextmenucls}
                    >
                        {buttonsRules.map((item, index) => (
                            <RightClickRules
                                startIndex={startIndex}
                                endIndex={endIndex}
                                openContext={openContext}
                                item={item}
                                selectedIdx={selectedIdx}
                                textObjects={textObjects}
                                updateDataEvent={updateDataEvent}
                            />
                        ))}
                    </ContextMenu>
                </ClickAwayListener>
            )}
        </Grid>
    );
};
export default Summary;
