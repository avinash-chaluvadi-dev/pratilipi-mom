import React, { useState, useEffect, useMemo, useRef } from "react";
import {
    Box,
    Typography,
    Divider,
    Grid,
    Button,
    Paper,
    TextField,
    ClickAwayListener,
    Menu,
} from "@material-ui/core";
import Select, { components } from "react-select";
import CircularProgress from "@mui/material/CircularProgress";
import { ContextMenu, ContextMenuTrigger } from "react-contextmenu";
import { Close, Info } from "@mui/icons-material";
import AutorenewIcon from "@mui/icons-material/Autorenew";
import useStyles from "screens/FeedBackLoop/styles";
import arrowIcon from "static/images/arrowIcon.svg";
import lineConnectorIcon from "static/images/summarylineconnector.svg";
import userIcon from "static/images/user.svg";
import filterIcon from "static/images/filter.svg";
import videoIcon from "static/images/video.svg";
import imageViewIcon from "static/images/imageView.svg";
import ReactTooltip from "react-tooltip";
import openPopupIcon from "static/images/expand.png";
import DropDown from "components/DropDown";
import AdvDropDown from "components/AdvDropDown";
import MMDatePicker from "components/MMDatePicker";
import Modal from "components/Modal";
import PreviewFile from "screens/FeedBackLoop/components/Preview";
import customStyles from "screens/FeedBackLoop/components/DetailedView/styles";
import { useDispatch, useSelector } from "react-redux";
import { updateFeedback } from "store/action/mom";
import ContentEditable from "react-contenteditable";
import TeamMember from "screens/FeedBackLoop/components/TeamMember";
import RightClickRules from "screens/FeedBackLoop/components/RightClickRules";
import NoDataFound from "screens/FeedBackLoop/components/NoDataFound";
import {
    FilterOptions,
    MakerdataOptions,
    LabelDataOptions,
    EntitiesDataOptions,
    SentimentDataOptions,
    ButtonsDataRules,
} from "screens/FeedBackLoop/masterData";
import {
    prepareArr,
    markSelections,
    getStartEndStart,
    loadAssignedValues,
    isArray,
} from "screens/FeedBackLoop/commonLogic";
import PopupOutView from "screens/FeedBackLoop/components/PopupOutView";
import select from "static/images/selectIcon.png";
import moment from "moment";
import { getPlayBackFilePath } from "store/action/upload";

const DetailedView = (props) => {
    const classes = useStyles();
    const dispatch = useDispatch();
    const tabsValue = useSelector((state) => state.tabsReducer);
    const { momStore, redirection_mask_id, isSaveChanges, meetingmetadata } =
        useSelector((state) => state.momReducer);
    // const dropDownReducerValue = useSelector((state) => state.dropDownReducer);
    // const { isLoader } = useSelector((state) => state.loaderReducer);
    let { playbackFileUrl } = useSelector((state) => state.uploadReducer);
    const [open, setOpen] = useState(false);

    let options = FilterOptions();
    let makerdata = MakerdataOptions();
    let labelOptions = LabelDataOptions();
    let entitiesOptions = EntitiesDataOptions();
    let sentimentOptions = SentimentDataOptions();
    const buttonsRules = ButtonsDataRules();

    let MoMStoreDataForFilter = momStore?.concatenated_view;
    let [MoMStoreData, setMoMStoreData] = useState(momStore?.concatenated_view);
    const [textObjects, setTextObjects] = useState({});
    const [summaryTextObjects, setSummaryTextObjects] = useState({});
    const [selectedRow, setSelectedRow] = useState();
    const [Transcript, setTranscript] = useState();
    const [isLoadBtn, setIsLoadBtn] = useState(true);
    const [selectedOption, setSelectedOption] = useState(null);
    const [selectedOption2, setSelectedOption2] = useState(null);
    const [isApplyFilter, setApplyFilter] = useState(true);
    const [appliedFilter, setAppliedFilter] = useState(false);
    const [filteredData, setFilteredData] = useState(false);
    const [count, setCount] = useState(0);
    const [tooltipOpen, setTooltipOpen] = useState(false);
    const [mouseEnterVal, setMouseEnterVal] = useState(false);
    const [mouseOverCardVal, setMouseOverCard] = useState(null);
    const [isOpenPopup, setOpenPopup] = useState(null);
    const [selectedRowData, setSelectedRowData] = useState({});
    const [openPreview, setOpenPreview] = useState(false);
    const [fileName, setFileName] = React.useState("");
    const [fileSize, setFileSize] = React.useState("");
    const [fileSource, setFileSource] = React.useState("");
    const [isImageGallaryOpen, setImageGallaryOpen] = useState(false);
    const [usernameValue, setUserName] = useState("");
    const [pageNum, setPageNum] = useState(1);
    const detailCls = customStyles();
    const [selectedIdx, setSelectedIdx] = useState(null);
    const [startIndex, setStartIndex] = useState();
    const [endIndex, setEndIndex] = useState();
    const [selectedText, setSelectedText] = useState();
    const [contextMenu, setContextMenu] = useState(null);
    const [isEnabled, setEnableDropDown] = useState(true);
    const [enabledDDValue, setEnabledDDValue] = useState(null);
    const [parentFilter, setParentFilter] = useState(null);
    const [oldTranscript, setOldTranscript] = useState();
    const [enableApplyFilterBtn, setEnableApplyFilterBtn] = useState(true);
    const [dateValue, setDateValue] = useState();
    const [MomDetails, setMomDetails] = useState(MoMStoreData || []);

    const [oldSummaryTextObjects, setOldSummaryTextObjects] = useState({});
    const [oldTextObjects, setOldTextObjects] = useState({});
    const [anchorEl, setAnchorEl] = useState(null);
    const [usernameModified, setUsernameModified] = useState(false);
    const transcirptWindow = useRef();
    const isCompleted =
        meetingmetadata.meeting_status.toLowerCase() === "completed"
            ? true
            : false;
    const sectionopen = Boolean(anchorEl);
    const openSection = (event) => {
        setAnchorEl(event.currentTarget);
    };
    const closeSection = () => {
        setAnchorEl(null);
    };
    const customSelectStyles = {
        singleValue: (provided) => ({
            ...provided,
            color: "#286ce2",
        }),
        placeholder: (provided) => ({
            ...provided,
            color: "#286ce2",
        }),
    };
    const DropdownIndicator = (props) => {
        return (
            <components.DropdownIndicator {...props}>
                <img
                    src={select}
                    alt="select"
                    // {...rest}
                    // className={clsx(className, classes.selectIcon)}
                />
            </components.DropdownIndicator>
        );
    };

    const updateJson = (arr) => {
        dispatch({
            type: "UPDATE_MOM_STORE",
            payload: { momStore: arr },
        });
    };

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

    const filterLabelData = (items, optionsArr, type) => {
        if (isArray(items)) {
            items = items ? items : [];
            let final = optionsArr;
            let result = [];
            result = final?.filter(function (item) {
                let values = item.value;
                // console.log('===acha==', items, item.value);
                // console.log('===', items?.indexOf(values));
                return items && items?.indexOf(values) > -1;
            });
            if (result.length > 0) {
                let res = [];
                for (var i = 0; i < result.length; i++) {
                    res.push({
                        label: result[i].labelWithoutIcon,
                        value: result[i].value,
                    });
                }
                result = res;
            }
            return result;
        }
    };

    const handleDateChange = (data, idx) => {
        data = moment(data, moment.defaultFormat).format("YYYY-MM-DD");
        setDateValue(data);
        MoMStoreData[idx].date = data;
        momStore.concatenated_view = MoMStoreData;
        updateJson(momStore);
        compareData();
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
        momStore["map_username"][speaker_id] = usernameValue[idx];
        momStore["map_username"][usernameValue[idx]] = usernameValue[idx];
        updateJson(momStore);
        hoverOn(idx);
        loadAssignToValues();
        if (tabsValue.from === "mom" && tabsValue.isUserSelection) {
            closeStatusBlockBar();
        }
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 2000);
        compareData();
    };

    const loadAssignToValues = () => {
        let data = loadAssignedValues(momStore);
        //console.log('=====okmama=====', data);
        updateJson(data);
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

    useEffect(() => {
        setFileName(momStore.file_name);
        setFileSize(momStore.size);
        dispatch({ type: "SWITCH_TABS", payload: { title: 1 } });
        let finalList = MoMStoreData.filter(
            (v, i, a) => a.findIndex((t) => t.speaker_id === v.speaker_id) === i
        );

        MoMStoreData?.map((item, idx) => {
            let { words, type } = item.entities[0];
            let data = getStartEndStart(item.transcript, words, type);
            let final = markSelections(item.transcript, data);
            MoMStoreData[idx].transcript = final;
        });
        momStore.concatenated_view = MoMStoreData;
        updateJson(momStore);

        MoMStoreDataForFilter?.map((item, idx) => {
            let { words, type } = item.entities[0];
            let data = getStartEndStart(item.transcript, words, type);
            let final = markSelections(item.transcript, data);
            MoMStoreDataForFilter[idx].transcript = final;
        });
        momStore.concatenated_view = MoMStoreDataForFilter;
        updateJson(momStore);
        loadAssignToValues();

        setTooltipOpen(true);
        LoadDefaultData();
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 1500);
    }, [dispatch]);

    const onFocus = (idx) => {
        setSelectedIdx(idx);
    };

    const handleChange = (selectedOption) => {
        setSelectedOption(selectedOption);
        setEnableDropDown(false);
        setSelectedOption2({});
        let result = [];
        if (selectedOption.value === "Labels") {
            result = labelOptions;
            setParentFilter("label");
        } else if (selectedOption.value === "Entities") {
            result = entitiesOptions;
            setParentFilter("entities");
        } else if (selectedOption.value === "Markers") {
            result = makerdata;
            setParentFilter("marker");
        } else if (selectedOption.value === "Sentiments") {
            result = sentimentOptions;
            setParentFilter("sentiment");
        } else if (selectedOption.value === "Participants") {
            result = momStore["AssignTo"];
            setParentFilter("Participants");
        }
        setEnabledDDValue(result);
    };

    const handleNormalSelect = (selectedOption) => {
        setEnableApplyFilterBtn(false);
        setSelectedOption2(selectedOption);
    };

    const handleChangeDropDownLabel = async (selectedOption, idx) => {
        // selectedOption.value = selectedOption.value.slice(0, -1);
        MoMStoreData[idx].label = MoMStoreData[idx].label
            ? MoMStoreData[idx].label
            : [];
        if (MoMStoreData[idx].label.indexOf(selectedOption.value) === -1) {
            MoMStoreData[idx].bkp_label = {
                ...MoMStoreData[idx].bkp_label,
                [selectedOption.value]: 100,
            };
            MoMStoreData[idx].label.push(selectedOption.value);
            momStore.concatenated_view = MoMStoreData;
            updateJson(momStore);
        } else {
            MoMStoreData[idx].label = MoMStoreData[idx].label.filter(
                (item) => item !== selectedOption.value
            );
            delete MoMStoreData[idx].bkp_label[selectedOption.value];
            momStore.concatenated_view = MoMStoreData;
            updateJson(momStore);
        }
        hoverOff();
        compareData();
    };

    const handleChangeDropDownEntity = (selectedOption, idx) => {
        // setSelectedOptionEntity({ ...selectedOption, [idx]: selectedOption });
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
        setStartIndex("");
        setEndIndex("");
    };

    const handleChangeDropDownSentiments = async (selectedOption, idx) => {
        // setSelectedOptionSentiments(selectedOption);
        //console.log('===acha===0', momStore, MoMStoreData);
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
            //console.log('===acha===', MoMStoreData);
            momStore.concatenated_view = MoMStoreData;
            //console.log('===acha===1', momStore);
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

    const applyFilter = async () => {
        setAnchorEl(null);
        setPageNum(1);
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        setOldTranscript(MoMStoreDataForFilter);

        if (parentFilter !== "Participants") {
            let arr = MoMStoreDataForFilter.filter(
                (item, idx) =>
                    item[parentFilter]?.indexOf(selectedOption2?.value) !== -1
            );
            setMoMStoreData(arr);
            if (arr.length <= 10) {
                fetchMoreData();
            }
            setCount(arr.length);
            setFilteredData(arr.length);
        }
        if (parentFilter === "Participants") {
            let arr = MoMStoreDataForFilter.filter((item, idx) => {
                let data =
                    momStore["map_username"] &&
                    momStore["map_username"][item["speaker_id"]];
                return data === selectedOption2?.value;
            });
            setMoMStoreData(arr);
            if (arr.length <= 10) {
                fetchMoreData();
            }
            setCount(arr.length);
            setFilteredData(arr.length);
        }
        if (parentFilter === "entities") {
            let arr = MoMStoreDataForFilter.filter(
                (item, idx) =>
                    item[parentFilter][0]?.type?.indexOf(
                        selectedOption2?.value
                    ) !== -1
            );
            setMoMStoreData(arr);
            if (arr.length <= 10) {
                fetchMoreData();
            }
            setCount(arr.length);
            setFilteredData(arr.length);
        }
        setApplyFilter(!isApplyFilter);
        setAppliedFilter(true);
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 500);
    };

    const clearFilter = () => {
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        setFilteredData(oldTranscript?.length);
        setMoMStoreData(MoMStoreDataForFilter);
        setApplyFilter(!isApplyFilter);
        setCount(0);
        setPageNum(1);
        setAppliedFilter(false);
        setSelectedOption(null);
        setSelectedOption2(null);
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 1000);
        closeSection();
    };

    const stringifyMomStore = JSON.stringify(MoMStoreDataForFilter);
    useEffect(() => {
        setFilteredData(oldTranscript?.length);
        setApplyFilter(true);
        setAppliedFilter(false);
        setFileSource(playbackFileUrl);
    }, [tabsValue, playbackFileUrl, filteredData, stringifyMomStore]);

    useEffect(() => {
        setMoMStoreData(momStore?.concatenated_view);
    }, []);

    const prepareFirstData = () => {
        MoMStoreData = MoMStoreData.map((item, idx) => {
            return {
                ...item,
                bkp_label: item.label,
                bkp_sentiment: item.sentiment,
                bkp_marker: item.marker,
                label: prepareArr(item.label),
                sentiment: prepareArr(item.sentiment),
                marker: prepareArr(item.marker),
            };
        });
        momStore.concatenated_view = MoMStoreData;
        updateJson(momStore);
    };

    const prepareFirstDataFilter = () => {
        MoMStoreDataForFilter = MoMStoreDataForFilter.map((item, idx) => {
            return {
                ...item,
                bkp_label: item.label,
                bkp_sentiment: item.sentiment,
                bkp_marker: item.marker,
                label: prepareArr(item.label),
                sentiment: prepareArr(item.sentiment),
                marker: prepareArr(item.marker),
            };
        });
        momStore.concatenated_view = MoMStoreDataForFilter;
        updateJson(momStore);
    };

    const saveChangesAPICall = (type, data) => {
        if (data) {
            MoMStoreData = data;
        }
        if (type === "bar") {
            dispatch({
                type: "SWITCH_TABS",
                payload: {
                    isUserSelection: false,
                    userName: "",
                    isHilightCard: false,
                    highlitedGroup: "",
                    from: "",
                },
            });
        }
        isCancelSaveChanges();
        let saveExternalUsers = momStore["ExternalParticipants"]
            ? momStore["ExternalParticipants"]
            : [];
        // sessionStorage.setItem(
        //     "saveExternalUsers",
        //     JSON.stringify(saveExternalUsers)
        // );
        MoMStoreData = MoMStoreData.map((item) => {
            return {
                ...item,
                speaker_id: momStore["map_username"][item.speaker_id],
                label: item.bkp_label || {},
                sentiment: item.bkp_sentiment ? item.bkp_sentiment : {},
                marker: item.bkp_marker ? item.bkp_marker : {},
                transcript: item.transcript.replace(/(<([^>]+)>)/gi, ""),
                assign_to: item.assign_to
                    ? momStore["map_username"][item.assign_to]
                        ? momStore["map_username"][item.assign_to]
                        : item.assign_to
                    : "",
            };
        });
        momStore.concatenated_view = MoMStoreData;
        dispatch(
            updateFeedback(
                redirection_mask_id ||
                    window?.location?.pathname?.split("/")?.splice(-1)[0],
                momStore
            )
        );
        loadAssignToValues();
    };

    useEffect(() => {
        props.childFunc.current = saveChangesAPICall;
    }, []);

    const hoverOn = (idx, rowObj, type) => {
        setMouseOverCard(idx);
        setMouseEnterVal(true);
        if (type) {
            setSelectedRow(rowObj);
            handleToggle(idx, type);
        }
    };

    const hoverOff = () => {
        setMouseEnterVal(false);
    };

    const openModal = (item, idx) => {
        setSelectedRowData({ item, idx });
        setImageGallaryOpen(true);
        setOpenPopup(true);
    };

    const handleModalClose = () => {
        setOpenPopup(false);
        if (isImageGallaryOpen) {
            setImageGallaryOpen(false);
        }
    };

    const handlePreviewOpen = (rowData) => {
        setOpenPreview(true);
        if (rowData.video_path) {
            dispatch(getPlayBackFilePath(rowData.video_path));
        }
    };

    const handlePreviewClose = (e) => {
        setOpenPreview(false);
    };

    const imageGallaryOpen = (item, idx) => {
        setImageGallaryOpen(true);
        setSelectedRowData({ item, idx });
        setOpenPopup(true);
    };

    const handleChangeTemp = (event) => {
        let final = { ...MoMStoreData };
        final[selectedIdx].transcript = event.target.value;
        setTranscript(MoMStoreData);
        setTextObjects({ ...textObjects, [selectedIdx]: event.target.value });
        isApplySaveChanges();
    };

    // const handleUserName = (value, item, idx) => {
    //     if (item?.startsWith("User 0") && item?.split("User 0").length > 1) {
    //         return value || item;
    //     }
    //     return value || "User 0" + item;
    // };

    const closeStatusBlockBar = (item, idx) => {
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        if (tabsValue.userName !== "" && tabsValue.isUserSelection) {
            setTranscript(oldTranscript);
        }
        dispatch({
            type: "SWITCH_TABS",
            payload: {
                isUserSelection: false,
                userName: "",
                isHilightCard: false,
                highlitedGroup: "",
                from: "",
            },
        });
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 1000);
    };

    // const ConfidenceScore = (item) => {
    //     if (item.confidence > "80") {
    //         return "confidencegreenscore";
    //     } else if (item.confidence > "60" && item.confidence < "80") {
    //         return "confidenceyellowscore";
    //     } else {
    //         return "confidenceredscore";
    //     }
    // };

    const onSummaryInputchange = (event, idx) => {
        let final = { ...MoMStoreData };
        final[selectedIdx].summary = event.target.value;
        setTranscript(MoMStoreData);

        setSummaryTextObjects({
            ...summaryTextObjects,
            [idx]: event.target.value,
        });
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

    const handleToggle = (idx, type) => {
        setSelectedIdx(idx);
        if (type === "transcript") {
            setOpen(true);
        }
    };

    useMemo(() => {
        prepareFirstData();
    }, []);

    useMemo(() => {
        prepareFirstDataFilter();
    }, []);

    const handleCloseMenu = () => {
        setOpen(false);
        setContextMenu(null);
    };

    const assignToChange = (id, value) => {
        MoMStoreData[id].assign_to = value;
        momStore.concatenated_view = [...MoMStoreData];
        updateJson(momStore);
    };

    const UpdateDataEvent = (textObjectsVal) => {
        setTextObjects(textObjectsVal);
        setSelectedText("");
        setStartIndex("");
        setEndIndex("");
        handleCloseMenu();
        isApplySaveChanges();
    };

    useEffect(() => {
        transcirptWindow.current.scrollTo(0, 0);
    }, [transcirptWindow, props.scrollReset]);

    const filterEntitiesData = (items, optionsArr) => {
        items = items ? items[0]?.type : [];
        let final = optionsArr;
        let result = [];
        result = final.filter(function (item) {
            let dataVal = item.value;
            dataVal = dataVal === "Name" ? "Person Name" : dataVal;
            return items.indexOf(dataVal) > -1;
        });
        if (result.length > 0) {
            let res = [];
            for (var i = 0; i < result.length; i++) {
                res.push({
                    label: result[i].labelWithoutIcon,
                    value: result[i].value,
                });
            }
            result = res;
        }
        //console.log('=====result=======', result);

        return result;
    };

    const mainComponent = (item, idx) => {
        return (
            <>
                <Grid item xs={6} className={classes.actionCardCss}>
                    <Paper
                        className={` ${
                            tabsValue.isHilightCard &&
                            tabsValue.highlitedGroup.includes(
                                item?.speaker_label
                            )
                                ? classes.borderCls
                                : ""
                        } ${classes.paperCommonCss} ${
                            mouseOverCardVal === idx && mouseEnterVal
                                ? classes.paperHover
                                : ""
                        } `}
                        onMouseEnter={() => hoverOn(idx)}
                        onMouseLeave={() => hoverOff(idx)}
                    >
                        <Grid
                            container
                            className={classes.rootGrid}
                            spacing={2}
                            // onMouseLeave={handleClose}
                        >
                            <Grid itemxs={8} className={classes.profileBarLeft}>
                                <img
                                    src={userIcon}
                                    height="20px"
                                    width="20px"
                                    alt=""
                                />

                                <TextField
                                    type="text"
                                    variant="outlined"
                                    size="small"
                                    disabled={isCompleted}
                                    className={detailCls.userText1}
                                    placeholder="User name"
                                    InputProps={{
                                        style: { width: "80px" },
                                    }}
                                    value={
                                        usernameModified
                                            ? usernameValue[idx]
                                            : momStore["map_username"] &&
                                              momStore["map_username"][
                                                  item?.speaker_id
                                              ]
                                    }
                                    onChange={(e) => {
                                        setUsernameModified(true);
                                        setUserName(
                                            {
                                                ...usernameValue,
                                                [idx]: e.target.value,
                                            }
                                            // idx
                                        );
                                    }}
                                    onBlur={(e) => {
                                        updateUserName(
                                            e,
                                            idx,
                                            "user02",
                                            item?.speaker_id
                                        );
                                        setUsernameModified(false);
                                    }}
                                />
                                <Typography className={classes.timer}>
                                    {item?.start_time}
                                    {" - "}
                                    {item?.end_time}
                                </Typography>
                                <ReactTooltip delayShow={500} />
                                {/* {mouseOverCardVal === idx && mouseEnterVal && ( */}
                                <img
                                    src={imageViewIcon}
                                    height="20px"
                                    width="20px"
                                    className={classes.videoIcon}
                                    data-tip
                                    data-for="keyframes"
                                    onClick={() => imageGallaryOpen(item, idx)}
                                    alt=""
                                />
                                <ReactTooltip
                                    id="keyframes"
                                    place="top"
                                    offset="{'bottom': 5, 'left': -5}"
                                    effect="solid"
                                    border
                                    textColor="#333333"
                                    backgroundColor="#ffffff"
                                    borderColor="#286ce2"
                                    className={classes.tooltip}
                                >
                                    Key Frames
                                </ReactTooltip>
                                <img
                                    src={videoIcon}
                                    height="20px"
                                    width="20px"
                                    className={classes.videoIcon}
                                    data-tip
                                    data-for="playvideo"
                                    onClick={() => handlePreviewOpen(item)}
                                    alt=""
                                />
                                <ReactTooltip
                                    id="playvideo"
                                    place="top"
                                    offset="{'bottom': 5, 'left': -5}"
                                    effect="solid"
                                    border
                                    textColor="#333333"
                                    backgroundColor="#ffffff"
                                    borderColor="#286ce2"
                                    className={classes.tooltip}
                                >
                                    Play Recording
                                </ReactTooltip>
                                {/* )} */}
                                {/* {mouseOverCardVal === idx && mouseEnterVal && ( */}
                                <DropDown
                                    disabled={isCompleted}
                                    value={
                                        filterEntitiesData(
                                            item.entities,
                                            entitiesOptions
                                        )?.length > 1
                                            ? [
                                                  {
                                                      label:
                                                          "Entities" +
                                                          `(${
                                                              filterEntitiesData(
                                                                  item.entities,
                                                                  entitiesOptions
                                                              )?.length
                                                          })`,
                                                      value:
                                                          "Entities" +
                                                          `(${
                                                              filterEntitiesData(
                                                                  item.entities,
                                                                  entitiesOptions
                                                              )?.length
                                                          })`,
                                                  },
                                                  ...filterEntitiesData(
                                                      item.entities,
                                                      entitiesOptions
                                                  ),
                                              ]
                                            : filterEntitiesData(
                                                  item.entities,
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
                                />
                            </Grid>

                            <div
                                onClick={() => handleToggle(idx, "transcript")}
                            >
                                <ContextMenuTrigger id="folder-context-menu">
                                    <ContentEditable
                                        html={textObjects[idx]}
                                        // innerRef={(elt) => (innerRef.current = elt)}
                                        // onBlur={handleBlur}
                                        onMouseDown={(evt) => {
                                            // evt.preventDefault(); // Avoids loosing focus from the editable area
                                            // document.execCommand(props.cmd, false, props.arg); // Send the command to the browser
                                        }}
                                        onChange={handleChangeTemp}
                                        disabled={isCompleted}
                                        // value={textObjects[idx]}
                                        onSelect={(e) =>
                                            handleSelectedText(e, idx)
                                        }
                                        onFocus={() => onFocus(idx)}
                                        onBlur={() => compareData(idx)}
                                        style={{
                                            height: "46px",
                                            overflowY: "auto",
                                            width: "535px",
                                            textAlign: "left",
                                            color: "#949494",
                                            fontSize: "12px",
                                            fontWeight: "bold",
                                            paddingLeft: "28px",
                                            paddingTop: "2px",
                                        }}
                                    />
                                </ContextMenuTrigger>
                            </div>
                        </Grid>
                        <Box component="div" display="flex">
                            {(mouseOverCardVal === idx && mouseEnterVal) ||
                            (tabsValue.isHilightCard &&
                                tabsValue.highlitedGroup.includes(
                                    item?.speaker_label
                                )) ? (
                                <img
                                    src={lineConnectorIcon}
                                    height="10px"
                                    width="150px"
                                    className={classes.connector}
                                    style={{ margin: "-29px 0px 0px 578px" }}
                                    alt=""
                                />
                            ) : (
                                <img
                                    src={arrowIcon}
                                    height="10px"
                                    width="10px"
                                    className={classes.connector}
                                    alt=""
                                />
                            )}
                        </Box>
                    </Paper>
                </Grid>

                <Grid item xs={6} className={classes.actionCardCss}>
                    {/* <ContextMenuTrigger id="contextmenu"> */}
                    <Paper
                        className={` ${
                            tabsValue.isHilightCard &&
                            tabsValue.highlitedGroup.includes(
                                item?.speaker_label
                            )
                                ? classes.borderCls
                                : ""
                        }  ${classes.paperCommonCssSecond}  ${
                            mouseOverCardVal === idx && mouseEnterVal
                                ? classes.paperHover
                                : ""
                        } `}
                        onMouseEnter={() => hoverOn(idx, item, "summary")}
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
                                className={classes.profileBarRight}
                            >
                                <Typography className={classes.timerRight}>
                                    {item?.start_time}
                                    {" - "}
                                    {item?.end_time}
                                </Typography>

                                {
                                    // <Tooltip title="Change Label" placement="top" open={tooltipOpen} onClick={handleTooltip} arrow>
                                    <DropDown
                                        disabled={isCompleted}
                                        value={
                                            filterLabelData(
                                                item?.label,
                                                labelOptions
                                            )?.length > 1
                                                ? [
                                                      {
                                                          label:
                                                              "Labels" +
                                                              `(${
                                                                  filterLabelData(
                                                                      item?.label,
                                                                      labelOptions
                                                                  )?.length
                                                              })`,
                                                          value:
                                                              "Labels" +
                                                              `(${
                                                                  filterLabelData(
                                                                      item?.label,
                                                                      labelOptions
                                                                  )?.length
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
                                        isMulti={true}
                                        isCount={true}
                                    />
                                }
                                <TeamMember
                                    type="Organizer"
                                    idx={idx}
                                    disabled={isCompleted}
                                    value={item?.assign_to}
                                    compareData={compareData}
                                    loadAssignToValues={loadAssignToValues}
                                    assignToChange={assignToChange}
                                />
                                <Grid style={{ width: "80px" }}>
                                    <MMDatePicker
                                        handleDateChange={(e) =>
                                            handleDateChange(e, idx)
                                        }
                                        idx={idx}
                                        disabled={isCompleted}
                                        customIcon={false}
                                        width={"10px"}
                                        height={"10px"}
                                        value={item?.date}
                                        placeholder="Date"
                                        type="DetailView"
                                    />
                                </Grid>
                                <DropDown
                                    disabled={isCompleted}
                                    value={
                                        filterLabelData(
                                            item?.sentiment,
                                            sentimentOptions,
                                            "sentiment"
                                        )?.length > 1
                                            ? [
                                                  {
                                                      label:
                                                          "sentiments" +
                                                          `(${
                                                              filterLabelData(
                                                                  item?.sentiment,
                                                                  sentimentOptions,
                                                                  "sentiment"
                                                              )?.length
                                                          })`,
                                                      value:
                                                          "sentiments" +
                                                          `(${
                                                              filterLabelData(
                                                                  item?.sentiment,
                                                                  sentimentOptions,
                                                                  "sentiment"
                                                              )?.length
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
                                disabled={isCompleted}
                                className={detailCls.userText}
                                onChange={(e) => {
                                    onSummaryInputchange(e, idx);
                                }}
                                InputProps={{
                                    classes: {
                                        input: classes.thaiTextFieldInputProps,
                                    },
                                    // style: { width: '550px' },
                                }}
                                onFocus={() => onFocus(idx)}
                                onBlur={() => compareData(idx)}
                            />
                        </Grid>
                    </Paper>
                    {/* </ContextMenuTrigger> */}
                </Grid>
            </>
        );
    };

    const isLoadBtnClick = () => {
        setIsLoadBtn(true);
    };

    const fetchMoreData = () => {
        setIsLoadBtn(false);
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        setPageNum(pageNum + 1);
        setTimeout(() => {
            setIsLoadBtn(true);
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 1000);
    };

    return (
        <Grid container alignItems="center">
            {!tabsValue.isUserSelection ? (
                <Grid container spacing={2}>
                    <Grid
                        container
                        item
                        xs={7}
                        direction="column"
                        className={detailCls.midblock}
                    >
                        <Button
                            variant="outlined"
                            className={classes.filterbtn}
                            onClick={openSection}
                            style={{ textTransform: "none" }}
                        >
                            Data Filter
                            <img
                                src={filterIcon}
                                height="20px"
                                width="20px"
                                alt=""
                            />
                        </Button>
                        <Grid item xs={12}>
                            <Menu
                                id="basic-menu"
                                anchorEl={anchorEl}
                                open={sectionopen}
                                onClose={closeSection}
                                style={{ top: "45px", overflow: "visible" }}
                                className={detailCls.menuWidth}
                            >
                                <Box
                                    component="form"
                                    sx={{
                                        display: "flex",
                                        margin: "20px",
                                        marginBottom: "30px",
                                        justifyContent: "space-between",
                                    }}
                                >
                                    <Select
                                        options={options}
                                        onChange={handleChange}
                                        value={selectedOption}
                                        placeholder={"Select"}
                                        className={classes.select}
                                        styles={customSelectStyles}
                                        components={{
                                            DropdownIndicator,
                                            IndicatorSeparator: () => null,
                                        }}
                                        isSearchable={false}
                                    ></Select>
                                    <Select
                                        options={enabledDDValue}
                                        onChange={handleNormalSelect}
                                        value={selectedOption2}
                                        placeholder={"Select"}
                                        className={classes.select}
                                        styles={customSelectStyles}
                                        components={{
                                            DropdownIndicator,
                                            IndicatorSeparator: () => null,
                                        }}
                                        isSearchable={false}
                                        disabled={isEnabled}
                                    ></Select>
                                </Box>
                                <Box
                                    component="div"
                                    sx={{
                                        display: "flex",
                                        flexWrap: "wrap",
                                        margin: "20px",
                                        justifyContent: "space-between",
                                    }}
                                >
                                    <Button
                                        color="secondary"
                                        onClick={clearFilter}
                                        startIcon={<AutorenewIcon />}
                                        style={{
                                            textTransform: "none",
                                            minWidth: "200px",
                                            fontSize: "14px",
                                            fontWeight: "bold",
                                            color: "#286ce2",
                                            backgroundColor: "#FFFFFF",
                                            justifyContent: "flex-start",
                                        }}
                                    >
                                        Reset All Filters
                                    </Button>
                                    <Button
                                        autoFocus
                                        onClick={closeSection}
                                        variant="contained"
                                        style={{
                                            textTransform: "none",
                                            minWidth: "150px",
                                            fontSize: "14px",
                                            fontWeight: "bold",
                                            color: "#286ce2",
                                            backgroundColor: "#FFFFFF",
                                            borderRadius: "8px",
                                        }}
                                    >
                                        Cancel
                                    </Button>
                                    <Button
                                        autoFocus
                                        onClick={applyFilter}
                                        variant="contained"
                                        color="primary"
                                        size="small"
                                        style={{
                                            textTransform: "none",
                                            minWidth: "150px",
                                            fontSize: "14px",
                                            color: "#FFFFFF",
                                            backgroundColor: "#1665DF",
                                            borderRadius: "8px",
                                        }}
                                    >
                                        Apply Filters
                                    </Button>
                                </Box>
                            </Menu>
                        </Grid>
                    </Grid>
                    <Grid
                        container
                        item
                        xs={5}
                        direction="column"
                        className={detailCls.midblockright}
                    >
                        {/* <AdvDropDown
                            value={null}
                            options={entitiesOptions}
                            type={"Entity"}
                            isNormal={false}
                            isSearchable={false}
                            placeholder={"Entities"}
                        /> */}
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={(e) => saveChangesAPICall("bar")}
                            className={classes.button}
                            style={{
                                textTransform: "none",
                                margin: "0 10px 0 10px",
                                float: "right",
                                width: "145px",
                                height: "30px",
                                borderRadius: "8px",
                                fontSize: "14px",
                                fontWeight: "bold",
                                fontFamily: "Lato",
                                color: "#FFFFFF",
                                backgroundColor:
                                    !isSaveChanges || isCompleted
                                        ? "#9bbaeb"
                                        : "#1665DF",
                            }}
                            disabled={!isSaveChanges || isCompleted}
                        >
                            Save Changes
                        </Button>
                        {/* </Box> */}
                    </Grid>
                </Grid>
            ) : (
                <Grid
                    container
                    spacing={2}
                    className={detailCls.selectionblock}
                >
                    <Grid
                        container
                        item
                        xs={1}
                        direction="column"
                        style={{ maxWidth: "3%" }}
                    >
                        <Info />
                    </Grid>
                    <Grid
                        container
                        item
                        xs={8}
                        direction="column"
                        style={{ minWidth: "83%" }}
                    >
                        {tabsValue.isHilightCard &&
                        tabsValue.isUserSelection &&
                        tabsValue.from !== "momActions" ? (
                            <Typography className={detailCls.selectioncls}>
                                {`Displaying all the “${tabsValue.userName}” in the Detailed View. Close to go back to the detailed view`}
                            </Typography>
                        ) : tabsValue.isHilightCard &&
                          tabsValue.from === "momActions" ? (
                            <Typography className={detailCls.selectioncls}>
                                {/* {`Displaying all the Chunk_id “${tabsValue.userName}” in the Detailed View. Close to go back to the detailed view`} */}
                                {`Displaying Selected Chunk in highlighted color`}
                            </Typography>
                        ) : (
                            tabsValue.from !== "momActions" && (
                                <Typography className={detailCls.selectioncls}>
                                    {`Displaying all “${tabsValue.userName}” transcripts. Close to go back to the detailed view`}
                                </Typography>
                            )
                        )}
                    </Grid>
                    <Grid
                        container
                        item
                        xs={1}
                        direction="column"
                        style={{
                            display: "flex",
                            flexDirection: "row",
                            minWidth: "14%",
                        }}
                    >
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={(e) => saveChangesAPICall("bar")}
                            className={classes.button}
                            style={{
                                textTransform: "none",
                                margin: "0 20px 0 0",
                            }}
                            disabled={!isSaveChanges || isCompleted}
                        >
                            Save changes
                        </Button>
                        <Close
                            onClick={closeStatusBlockBar}
                            className={detailCls.pontercls}
                        />
                    </Grid>
                </Grid>
            )}
            <Grid
                container
                style={{
                    position: "relative",
                }}
                spacing={3}
            >
                <Grid item xs={6}>
                    <Paper className={classes.paperCommonHead}>
                        Transcript
                    </Paper>
                </Grid>
                <Grid item xs={6}>
                    <Paper className={classes.paperCommonHeadSecond}>
                        Transcript Summary
                    </Paper>
                </Grid>
                <Grid
                    container
                    spacing={3}
                    xs={12}
                    ref={transcirptWindow}
                    className={classes.paperFullWidth}
                >
                    {MoMStoreData.length > 0 &&
                        MoMStoreData?.slice(0, 10 * pageNum)?.map(
                            (item, idx) => {
                                return (
                                    <>
                                        {!appliedFilter &&
                                            tabsValue.from === "mom" &&
                                            momStore["map_username"] &&
                                            momStore["map_username"][
                                                item?.speaker_id
                                            ] === tabsValue.userName && (
                                                <>{mainComponent(item, idx)}</>
                                            )}

                                        {(tabsValue.from === "" ||
                                            tabsValue.from === "summary" ||
                                            tabsValue.from === "momActions") &&
                                            !appliedFilter && (
                                                <>{mainComponent(item, idx)}</>
                                            )}

                                        {appliedFilter &&
                                            parentFilter !== "Participants" &&
                                            item[parentFilter]?.indexOf(
                                                selectedOption2?.value
                                            ) !== -1 && (
                                                <>{mainComponent(item, idx)}</>
                                            )}
                                        {appliedFilter &&
                                            parentFilter === "Participants" &&
                                            momStore["map_username"] &&
                                            momStore["map_username"][
                                                item["speaker_id"]
                                            ] === selectedOption2?.value && (
                                                <>{mainComponent(item, idx)}</>
                                            )}
                                    </>
                                );
                            }
                        )}

                    {MoMStoreData.length === 0 && (
                        <>
                            <NoDataFound
                                margin={"15rem 2rem 0 0rem"}
                                size={6}
                            />
                            <NoDataFound
                                margin={"15rem 0rem 0 5rem"}
                                size={6}
                            />
                        </>
                    )}
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
                                filterEntitiesData={filterEntitiesData}
                                entitiesOptions={entitiesOptions}
                                handleChangeDropDownLabel={
                                    handleChangeDropDownLabel
                                }
                                handleDateChange={handleDateChange}
                                sentimentOptions={sentimentOptions}
                                handleChangeDropDownEntity={
                                    handleChangeDropDownEntity
                                }
                                handleChangeDropDownSentiments={
                                    handleChangeDropDownSentiments
                                }
                                summaryTextObjects={summaryTextObjects}
                                textObjects={textObjects}
                                onSummaryInputchange={onSummaryInputchange}
                                isImageGallaryOpen={isImageGallaryOpen}
                                TeamMember={TeamMember}
                                updateUserName={updateUserName}
                                disabled={isCompleted}
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
                <Grid item xs={6}></Grid>
                <Grid item xs={6}>
                    {count === 0 && MoMStoreData.length > pageNum * 10 && (
                        <h4
                            style={{
                                textAlign: "right",
                                color: "#372EC1",
                                cursor: "pointer",
                                paddingRight: "31px",
                                margin: "12px 0 0 0",
                                fontWeight: "initial",
                            }}
                            onClick={fetchMoreData}
                        >
                            {` Showing ${
                                pageNum * 10 > MoMStoreData.length
                                    ? MoMStoreData.length
                                    : pageNum * 10
                            } of ${MoMStoreData.length} transcripts`}{" "}
                            {isLoadBtn && (
                                <Button
                                    variant={"outlined"}
                                    className={classes.btnColor}
                                >
                                    Load More...
                                </Button>
                            )}
                            {!isLoadBtn && (
                                <Button
                                    className={classes.btnColorGreen}
                                    onClick={isLoadBtnClick}
                                    loading={true}
                                    endIcon={
                                        <CircularProgress
                                            style={{
                                                width: "15px",
                                                height: "15px",
                                                color: "red",
                                            }}
                                        />
                                    }
                                    loadingPosition="end"
                                    variant="outlined"
                                >
                                    Load More
                                </Button>
                            )}
                        </h4>
                    )}
                    {count > 0 && count > pageNum * 10 && (
                        <h4
                            style={{
                                textAlign: "right",
                                color: "#372EC1",
                                cursor: "pointer",
                                paddingRight: "31px",
                                margin: "12px 0 0 0",
                                fontWeight: "initial",
                            }}
                            onClick={fetchMoreData}
                        >
                            {` Showing ${
                                pageNum * 10 > count ? count : pageNum * 10
                            } of ${count} transcripts`}{" "}
                            {isLoadBtn && (
                                <Button
                                    variant={"outlined"}
                                    className={classes.btnColor}
                                >
                                    Load More...
                                </Button>
                            )}
                            {!isLoadBtn && (
                                <Button
                                    className={classes.btnColorGreen}
                                    onClick={isLoadBtnClick}
                                    loading={true}
                                    endIcon={
                                        <CircularProgress
                                            style={{
                                                width: "15px",
                                                height: "15px",
                                                color: "red",
                                            }}
                                        />
                                    }
                                    loadingPosition="end"
                                    variant="outlined"
                                >
                                    Load More
                                </Button>
                            )}
                        </h4>
                    )}
                </Grid>

                {open && (
                    <ClickAwayListener onClickAway={handleCloseMenu}>
                        <ContextMenu
                            id="folder-context-menu"
                            className={detailCls.contextmenucls}
                        >
                            {buttonsRules.map((item, index) => (
                                <RightClickRules
                                    updateDataEvent={UpdateDataEvent}
                                    startIndex={startIndex}
                                    endIndex={endIndex}
                                    openContext={open}
                                    item={item}
                                    selectedIdx={selectedIdx}
                                    textObjects={textObjects}
                                />
                            ))}
                        </ContextMenu>
                    </ClickAwayListener>
                )}
            </Grid>
        </Grid>
    );
};
export default DetailedView;
