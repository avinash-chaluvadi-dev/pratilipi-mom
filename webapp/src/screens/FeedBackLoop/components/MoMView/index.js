import React, { useState, useEffect } from "react";
import { format } from "date-fns";
import {
    Box,
    Typography,
    Divider,
    Grid,
    MenuItem,
    TextField,
    Button,
    Paper,
    Menu,
    Checkbox,
    FormControl,
    IconButton,
    ListItemSecondaryAction,
} from "@material-ui/core";
import Select, { components } from "react-select";
import customStyles from "screens/FeedBackLoop/components/MoMView/useStyles";
import dropDownIcon from "static/images/dropDown.svg";
import CloseDropDown from "static/images/CloseDropDown.svg";
import {
    OutlinedInput,
    List,
    ListItem,
    ListItemText,
    Collapse,
} from "@mui/material";
import SelectUI from "@material-ui/core/Select";
import useStyles from "screens/FeedBackLoop/styles";
import MaterialTable from "material-table";
import {
    ExpandLess,
    ExpandMore,
    AddCircleOutline,
    NoEncryption,
} from "@mui/icons-material";
import MMDatePicker from "components/MMDatePicker";
import editIcon from "static/images/editIcon.svg";
import userIcon from "static/images/user.svg";
import collapse from "static/images/collapse.svg";
import Modal from "components/Modal";
import { useDispatch, useSelector } from "react-redux";
import lineConnectorIcon from "static/images/summarylineconnector.svg";
import TextInput from "components/TextInput";
import select from "static/images/selectIcon.png";
import { getMom, patchMeetingMetaData } from "store/action/mom";
// import { patchMeetingMetaData, getMeetingMetaData } from "store/action/mom";
// import { getScrumTeamName, addScrumTeamName } from "store/action/scrumTeam";
import moment from "moment";
import { updateFeedback } from "store/action/mom";
import ReactTooltip from "react-tooltip";
import { isEmpty } from "utils/utility";

const section = [
    "Action",
    "Action with Deadline",
    "Announcement",
    "Appreciation",
    "Escalation",
];

const MomView = () => {
    const classes = useStyles();
    const momCls = customStyles();
    const dispatch = useDispatch();
    const { meetingmetadata, redirection_mask_id, momStore } = useSelector(
        (state) => state.momReducer
    );
    const { scrumTeams } = useSelector((state) => state.scrumTeamNameReducer);
    let TranscriptData = momStore?.concatenated_view || [];
    let MomEntries = momStore?.mom_entries || {};
    const [Transcript] = useState(TranscriptData);
    const [open, setOpen] = useState({});
    const [selectedIdx, setSelectedIdx] = useState({});
    const [selectedSectionsData, setSelectedSections] = useState({});
    const [anchorEl, setAnchorEl] = useState(null);
    const [checked, setChecked] = useState({});
    const sectionopen = Boolean(anchorEl);
    const [dropDownOpen, setdropDownOpen] = useState({});
    const [openAddTeamMem, setOpenAddTeamMem] = useState({});
    const [TeamName, setTeamName] = useState({});
    // const [SubmitTeamData, setSubmitTeamData] = useState({});
    const [isIconVisibile, setIconVisibile] = useState(false);
    const [isOpenPopup, setOpenPopup] = useState(false);
    const [popupcontent, setOpenPopupContent] = useState(null);
    const [addTopicVal, setAddTopic] = useState([{}]);
    const [actionsOwner, setActionsOwner] = useState("");
    const [actionsDescription, setActionsDescription] = useState("");
    const [removeItem, setRemoveItem] = useState(false);
    const [getData, setGetData] = useState(MomEntries);
    const [addActionsArr, setAddActionsArr] = useState([{}]);
    const [projectname, setProjectName] = useState();
    const [organizer, setOrganizername] = useState("");
    const [attendees, setAttendees] = useState([]);
    const [locationValue, setLocation] = useState();
    const [meetingDuration, setMeetingDuration] = useState();
    const [uploadDate, setUploadDate] = useState(meetingmetadata?.meeting_date);
    const [dateValue, setDateValue] = useState(
        meetingmetadata?.mom_generation_date
    );
    const [teamMemberVal, setTeamMember] = useState({});
    const [newTranscriptSummary, setNewTranscriptSummary] = useState("");
    const [newTranscriptOwner, setNewTranscriptOwner] = useState("");
    const [newTranscriptDate, setNewTranscriptDate] = useState(null);
    const [transcriptModified, setTranscriptModified] = useState(false);
    const isCompleted =
        meetingmetadata.meeting_status.toLowerCase() === "completed"
            ? true
            : false;
    const attendeesStringfy = JSON.stringify(attendees);

    const serialNoColumn = {
        title: "SR.NO",
        field: "id",
        cellStyle: {
            borderRight: "1px solid #D6D4D4",
            textAlign: "center",
            color: "#333333",
            fontSize: "16px",
            fontFamily: "Lato",
            borderBottom: "0px",
            padding: "10px",
        },
        headerStyle: {
            borderRight: "1px solid #D6D4D4",
            textAlign: "center",
            color: "#666666",
            fontSize: "14px",
            fontWeight: "bold",
            fontFamily: "Lato",
            borderBottom: "0px",
            padding: "10px",
        },
        width: "10%",
    };

    const actionColumn = {
        title: "ACTIONS",
        field: "summary",
        render: (rowData) => {
            return (
                <Box onClick={(e) => handelDescOnTable(rowData)}>
                    {rowData.summary}
                </Box>
            );
        },
        cellStyle: {
            borderRight: "1px solid #D6D4D4",
            color: "#333333",
            fontSize: "16px",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
        headerStyle: {
            borderRight: "1px solid #D6D4D4",
            textAlign: "center",
            color: "#666666",
            fontSize: "14px",
            fontWeight: "bold",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
    };

    const escalationColumn = {
        title: "ESCALATIONS",
        field: "summary",
        render: (rowData) => {
            return (
                <Box onClick={(e) => handelDescOnTable(rowData)}>
                    {rowData.summary}
                </Box>
            );
        },
        cellStyle: {
            borderRight: "1px solid #D6D4D4",
            color: "#333333",
            fontSize: "16px",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
        headerStyle: {
            borderRight: "1px solid #D6D4D4",
            textAlign: "center",
            color: "#666666",
            fontSize: "14px",
            fontWeight: "bold",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
    };

    const actionWDColumn = {
        title: "ACTION ITEMS WITH DEADLINES",
        field: "summary",
        render: (rowData) => {
            return (
                <Box onClick={(e) => handelDescOnTable(rowData)}>
                    {rowData.summary}
                </Box>
            );
        },
        cellStyle: {
            borderRight: "1px solid #D6D4D4",
            color: "#333333",
            fontSize: "16px",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
        headerStyle: {
            borderRight: "1px solid #D6D4D4",
            textAlign: "center",
            color: "#666666",
            fontSize: "14px",
            fontWeight: "bold",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
    };

    const appreciationColumn = {
        title: "APPRECIATIONS",
        field: "summary",
        render: (rowData) => {
            return (
                <Box onClick={(e) => handelDescOnTable(rowData)}>
                    {rowData.summary}
                </Box>
            );
        },
        cellStyle: {
            borderRight: "1px solid #D6D4D4",
            color: "#333333",
            fontSize: "16px",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
        headerStyle: {
            borderRight: "1px solid #D6D4D4",
            textAlign: "center",
            color: "#666666",
            fontSize: "14px",
            fontWeight: "bold",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
    };

    const announcementColumn = {
        title: "ANNOUNCEMENTS",
        field: "summary",
        render: (rowData) => {
            return (
                <Box onClick={(e) => handelDescOnTable(rowData)}>
                    {rowData.summary}
                </Box>
            );
        },
        cellStyle: {
            borderRight: "1px solid #D6D4D4",
            color: "#333333",
            fontSize: "16px",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
        headerStyle: {
            borderRight: "1px solid #D6D4D4",
            textAlign: "center",
            color: "#666666",
            fontSize: "14px",
            fontWeight: "bold",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
    };

    const dateColumn = {
        title: "DATE",
        field: "date",
        render: (rowData) => {
            let date = rowData.date
                ? moment(rowData.date, moment.defaultFormat).format(
                      "DD MMM YYYY"
                  )
                : "";
            return <Box>{date}</Box>;
        },
        cellStyle: {
            borderRight: "1px solid #D6D4D4",
            textAlign: "center",
            color: "#333333",
            fontSize: "16px",
            fontFamily: "Lato",
            cursor: "pointer",
            width: "20%",
            borderBottom: "0px",
            padding: "10px",
        },
        headerStyle: {
            borderRight: "1px solid #D6D4D4",
            textAlign: "center",
            color: "#666666",
            fontSize: "14px",
            fontWeight: "bold",
            fontFamily: "Lato",
            cursor: "pointer",
            borderBottom: "0px",
            padding: "10px",
        },
    };

    const ownerColumn = {
        title: "OWNER",
        field: "assign_to",
        type: "numeric",
        render: (rowData) => {
            return (
                <Box>
                    {momStore["map_username"]
                        ? momStore["map_username"][rowData?.assign_to] ||
                          rowData?.assign_to
                        : rowData?.assign_to}
                </Box>
            );
        },
        cellStyle: {
            textAlign: "center",
            color: "#333333",
            fontSize: "16px",
            fontFamily: "Lato",
            width: "20%",
            borderBottom: "0px",
            padding: "10px",
        },
        headerStyle: {
            textAlign: "center",
            color: "#666666",
            fontSize: "14px",
            fontWeight: "bold",
            fontFamily: "Lato",
            borderBottom: "0px",
            padding: "10px",
        },
    };
    const momStoreStringify = JSON.stringify(momStore?.concatenated_view);
    const momEntriesStringify = JSON.stringify(MomEntries);
    useEffect(() => {
        setProjectName(meetingmetadata?.project_name);
        setOrganizername(meetingmetadata?.organiser);
        setLocation(meetingmetadata?.location);
        setMeetingDuration(meetingmetadata?.meeting_duration);
        setDateValue(meetingmetadata?.mom_generation_date);
        setTeamMember({
            Scrumteamname: meetingmetadata?.meeting?.full_team_name,
        });
        // dispatch({ type: 'ADD_ACTION', payload: [{}] });
        setGetData(MomEntries);
        let finalList = Transcript.filter(
            (v, i, a) => a.findIndex((t) => t.speaker_id === v.speaker_id) === i
        );
        setAttendees(finalList);
    }, [
        dispatch,
        momStore.request_id,
        momStoreStringify,
        meetingmetadata.meeting.full_team_name,
        momEntriesStringify,
    ]);

    useEffect(() => {
        setProjectName(meetingmetadata?.project_name);
        setOrganizername(meetingmetadata?.organiser);
        setLocation(meetingmetadata?.location);
        setMeetingDuration(meetingmetadata?.meeting_duration);
        setDateValue(meetingmetadata?.mom_generation_date);
        setTeamMember({
            Scrumteamname: meetingmetadata?.meeting?.full_team_name,
        });
    }, [scrumTeams]);

    useEffect(() => {
        // Updatiing the attendee details to the PDF
        if (!isEmpty(attendees)) {
            let resultAttendees = attendees.map(
                (attendee) => attendee.speaker_id
            );
            dispatch(
                patchMeetingMetaData(momStore.request_id, {
                    attendees: resultAttendees.toString(),
                })
            );
        }
    }, [dispatch, momStore.request_id, attendeesStringfy]);

    // const addNewMemer = (e, type) => {
    //     setOpenAddTeamMem({ ...openAddTeamMem, [type]: true });
    // };

    useEffect(() => {
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        dispatch(getMom(momStore?.request_id));
        setTimeout(() => {
            dispatch({
                type: "STOP_LOADER",
                payload: { isLoader: false },
            });
        }, 1500);
    }, [dispatch, transcriptModified]);

    const handleDropDownChange = (event, type) => {
        let teamval = scrumTeams.filter(
            (entry) => entry.name === `${event.target.value}`
        );
        setTeamMember({ ...teamMemberVal, [type]: event.target.value });

        if (event.target.value) {
            dispatch(
                patchMeetingMetaData(momStore.request_id, {
                    meeting: {
                        team_name: teamval[0].id,
                    },
                })
            );
        }
    };
    // const handleDrawerClose = (type) => {
    //     setOpenAddTeamMem({ ...openAddTeamMem, [type]: false });
    // };
    const handleDropDownClose = (type) => {
        setdropDownOpen({ [type]: false });
        setIconVisibile(false);
    };

    const handleDropDownOpen = (type) => {
        setdropDownOpen({ [type]: true });
        setIconVisibile(true);
    };

    // const handleAddTeam = (e, type) => {
    //     setTeamName({ ...TeamName, [type]: e.target.value });
    // };

    // const CancelTeamBtn = (type) => {
    //     setSubmitTeamData({ ...SubmitTeamData, [type]: false });
    //     setTeamName({ ...TeamName, [type]: "" });
    //     setOpenAddTeamMem({ ...openAddTeamMem, [type]: false });
    // };

    // const AddTeamBtn = (e, type) => {
    //     dispatch(addScrumTeamName({ name: TeamName["Scrumteamname"] }));
    //     setSubmitTeamData({ ...SubmitTeamData, [type]: false });
    //     setTeamName({ ...TeamName, [type]: "" });
    //     setOpenAddTeamMem({ ...openAddTeamMem, [type]: false });
    //     dispatch(getScrumTeamName());
    //     dispatch(getMeetingMetaData(redirection_mask_id));
    // };

    let assignTo = scrumTeams.map((item, idx) => {
        return {
            ...item,
            value: item.name,
            label: (
                <div className={classes.textOverflow}>
                    <img
                        src={userIcon}
                        height="12px"
                        width="12px"
                        pt={2}
                        className={classes.imgPadding}
                        alt=""
                    />
                    <span className={classes.spanMargin}>{item.name}</span>
                </div>
            ),
            icon: editIcon,
        };
    });

    const openItemsPopup = (doc, title) => {
        setRemoveItem(false);
        setOpenPopupContent({ ...doc, title });
        setOpenPopup(true);
    };

    const handleModalClose = () => {
        setOpenPopup(false);
    };

    const addTopic = async () => {
        if (popupcontent?.title === "Topics Discussed") {
            let dataArr = addTopicVal;
            dataArr.push({});
            await setAddTopic(dataArr);
            dispatch({ type: "ADD_ACTION", payload: dataArr });
        } else {
            let data = addActionsArr;
            data.push({});
            await setAddActionsArr(data);
            await dispatch({ type: "ADD_ACTION", payload: data });
        }
    };

    const handleActionDes = (event) => {
        setActionsDescription(event.target.value);
    };

    const handleActionOwner = (event) => {
        setActionsOwner(event.target.value);
    };

    const handelDescOnTable = (Data) => {
        dispatch({
            type: "SWITCH_TABS",
            payload: {
                title: 1,
                isHilightCard: true,
                userName: [Data?.speaker_label],
                from: "momActions",
                highlitedGroup: [Data?.speaker_label],
                isUserSelection: true,
            },
        });
    };

    const updateActions = (e, type, idx) => {
        addActionsArr[idx][type] = e.target.value;
        setAddActionsArr(addActionsArr);
    };

    const content = (
        <>
            <Box component="div" className={momCls.scrollHeight}>
                {(popupcontent?.title === "Others" ||
                    popupcontent?.title === "Topics Discussed") &&
                    addTopicVal.map(() => (
                        <Box component="div" display="flex">
                            <Checkbox
                                checked={checked}
                                inputProps={{ "aria-label": "controlled" }}
                                className={momCls.checkboxcls}
                            />
                            <Grid container className={momCls.contentbody}>
                                <Grid
                                    container
                                    item
                                    xs={6}
                                    direction="column"
                                    className={momCls.popuptextboxleft}
                                >
                                    <TextField
                                        required
                                        type="text"
                                        margin="0"
                                        variant="outlined"
                                        size={"small"}
                                        fullWidth
                                        placeholder={"Type your topic here..."}
                                        disabled={isCompleted}
                                        value={actionsDescription}
                                        className={momCls.textbox}
                                        onChange={handleActionDes}
                                    />
                                </Grid>
                                <Divider orientation="vertical" flexItem />
                                <Grid container item xs={3} direction="column">
                                    <TextField
                                        required
                                        type="text"
                                        margin="0"
                                        variant="outlined"
                                        size={"small"}
                                        fullWidth
                                        value={actionsOwner}
                                        placeholder={"Owner"}
                                        className={`${momCls.textbox} ${momCls.fullwithcls}`}
                                        onChange={handleActionOwner}
                                    />
                                </Grid>
                            </Grid>
                        </Box>
                    ))}
                {popupcontent?.title !== "Others" &&
                    popupcontent?.title !== "Topics Discussed" && (
                        <>
                            <Box
                                component="div"
                                display="flex"
                                style={{ paddingLeft: "15px" }}
                            >
                                <Grid item className={momCls.actionCardCss}>
                                    <TextField
                                        required
                                        multiline
                                        rows={5}
                                        style={{ width: "900px" }}
                                        margin="0"
                                        variant="outlined"
                                        placeholder={"Type Summary"}
                                        onChange={(e) =>
                                            setNewTranscriptSummary(
                                                e.target.value
                                            )
                                        }
                                    />
                                </Grid>
                            </Box>
                            <Grid
                                container
                                style={{
                                    paddingLeft: "30px",
                                    marginTop: "10px",
                                }}
                            >
                                <Grid column xs={3}>
                                    <Grid item>
                                        <Typography
                                            className={momCls.modalOwnerName}
                                        >
                                            Owner
                                        </Typography>
                                    </Grid>
                                    <TextField
                                        type="text"
                                        fullWidth
                                        margin="0"
                                        variant="outlined"
                                        size={"small"}
                                        placeholder={"Owner Name"}
                                        className={momCls.projectTextcls}
                                        onChange={(e) =>
                                            setNewTranscriptOwner(
                                                e.target.value
                                            )
                                        }
                                    />
                                </Grid>
                                <Grid
                                    column
                                    xs={2}
                                    style={{ marginLeft: "20px" }}
                                >
                                    <Grid item>
                                        <Typography
                                            className={momCls.modalOwnerName}
                                        >
                                            Date
                                        </Typography>
                                    </Grid>
                                    <Grid item>
                                        <MMDatePicker
                                            customIcon={true}
                                            handleDateChange={(e) =>
                                                setNewTranscriptDate(e)
                                            }
                                            placeholder="Enter Date"
                                        />
                                    </Grid>
                                </Grid>
                            </Grid>
                        </>
                    )}
            </Box>
        </>
    );

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

    const openSection = (event) => {
        setAnchorEl(event.currentTarget);
    };
    const closeSection = () => {
        setAnchorEl(null);
    };

    const seletedSections = (value, idx) => {
        if (checked[idx]) {
            setChecked({ ...checked, [idx]: false });
        } else {
            setChecked({ ...checked, [idx]: true });
        }
        setSelectedSections({
            [value]: [],
        });
    };

    const addSection = () => {
        let dataVal = getData;
        dataVal = { ...dataVal, ...selectedSectionsData };
        MomEntries = Object.assign(MomEntries, selectedSectionsData);
        console.log(MomEntries, dataVal);
        setGetData(dataVal);
        setAnchorEl(null);
    };

    const addActionsList = () => {
        let arr = [...getData[popupcontent.title], ...addActionsArr];
        getData[popupcontent.title] = arr;
        let tempTranshcriptDate =
            newTranscriptDate === "" ||
            newTranscriptDate === null ||
            newTranscriptDate === undefined
                ? ""
                : format(newTranscriptDate, "yyyy-MM-dd");
        setTranscriptModified(!transcriptModified);
        let tempDispatchData = {
            label: { [popupcontent.title]: 100 },
            manual_add: "true",
            summary: newTranscriptSummary,
            date: tempTranshcriptDate,
            assign_to: newTranscriptOwner,
        };
        dispatch(
            updateFeedback(
                redirection_mask_id ||
                    window?.location?.pathname?.split("/")?.splice(-1)[0],
                tempDispatchData
            )
        );
        setNewTranscriptDate(null);
        setNewTranscriptOwner("");
        setNewTranscriptSummary("");
        setGetData(getData);
        setOpenPopup(false);
        setAddActionsArr([{}]);
    };

    const closeActionPopup = () => {
        setAddActionsArr([{}]);
        setOpenPopup(false);
    };

    const deleteActionPopup = () => {
        setOpenPopup(false);
        section.map((item, idx) => {
            if (Object.keys(getData).map((ele) => ele === item))
                checked[idx] = false;
        });
        setTranscriptModified(!transcriptModified);
        let tempDispatchData = {
            manual_remove: "true",
            label: popupcontent.title,
        };
        if (getData[popupcontent.title].length !== 0) {
            dispatch(
                updateFeedback(
                    redirection_mask_id ||
                        window?.location?.pathname?.split("/")?.splice(-1)[0],
                    tempDispatchData
                )
            );
        }
        delete selectedSectionsData[popupcontent.title];
        delete getData[popupcontent.title];
        setGetData(getData);
    };

    const removeListItem = (doc, index, title) => {
        setOpenPopupContent({
            Name: "Are you sure you want to remove “Actions” section?",
            index: index,
            title: title,
        });
        setRemoveItem(true);
        setOpenPopup(true);
    };

    const EditData = (e, type, team) => {
        setOpenAddTeamMem({ ...openAddTeamMem, [type]: true });
        setdropDownOpen({ [type]: false });
        setIconVisibile(false);
        setTeamName({ ...TeamName, [type]: team.value });
    };

    const onchangeTextValue = (e, key) => {
        setProjectName(e.target.value);
    };

    const onchangeOrganiser = (e, key) => {
        setOrganizername(e.target.value);
    };

    const onchangeLocation = (e, key) => {
        setLocation(e.target.value);
    };

    const onchangeMeetinduration = (e, key) => {
        setMeetingDuration(e.target.value);
    };

    const onBlurField = (e, key, value) => {
        dispatch(patchMeetingMetaData(momStore.request_id, { [key]: value }));
    };

    const handleDateChange = (data, type) => {
        if (type === "mom_generation_date") {
            setDateValue(data);
        } else {
            setUploadDate(data);
        }
        let urlId = window.location.pathname.split("/");
        let Id = urlId[urlId.length - 1];
        dispatch(
            patchMeetingMetaData(momStore.request_id || Id, {
                [type]: data && format(data, "yyyy-MM-dd"),
            })
        );
    };

    const populateColumn = (title) => {
        if (title === "Action")
            return [serialNoColumn, actionColumn, ownerColumn];
        else if (title === "Action with Deadline")
            return [serialNoColumn, actionWDColumn, dateColumn, ownerColumn];
        else if (title === "Announcement")
            return [
                serialNoColumn,
                announcementColumn,
                dateColumn,
                ownerColumn,
            ];
        else if (title === "Appreciation")
            return [
                serialNoColumn,
                appreciationColumn,
                dateColumn,
                ownerColumn,
            ];
        else if (title === "Escalation")
            return [serialNoColumn, escalationColumn, dateColumn, ownerColumn];
    };
    const CustomizedListItem = (props) => {
        let { doc, index, title } = props;
        doc = doc.map((item, idx) => {
            return { ...item, id: idx + 1 };
        });
        const label =
            title === "Update"
                ? "Announcements   "
                : title === "Deadline"
                ? "Actions With Deadline  "
                : title;
        return (
            <>
                <ListItem button className={momCls.collapsecls} key={index}>
                    <img
                        src={collapse}
                        alt="Upload"
                        data-tip
                        data-for="collapse"
                        className={`${isCompleted ? momCls.removeSection : ""}`}
                        onClick={() => removeListItem(doc, index, title)}
                    />
                    <ReactTooltip
                        id="collapse"
                        place="top"
                        effect="solid"
                        border
                        textColor="#333333"
                        backgroundColor="#ffffff"
                        borderColor="#286ce2"
                        borderRadius="8px"
                        style={{
                            width: "10px",
                            fontSize: "10px!important",
                            fontWeight: "600!important",
                            fontFamily: "Lato!important",
                            borderColor: "#286ce2!important",
                            borderRadius: "10px!important",
                        }}
                    >
                        Remove Section
                    </ReactTooltip>
                    {"  "}
                    <ListItemText
                        primary={`${label} (${doc.length})`}
                        className={momCls.collapse_name}
                    />
                    <Typography
                        className={`${
                            isCompleted
                                ? momCls.collapse_sub_title_disabled
                                : momCls.collapse_sub_title
                        }`}
                        onClick={() => openItemsPopup(doc, title)}
                    >
                        {`Add  ${label}`}
                    </Typography>
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
                        <Box
                            sx={{
                                display: "flex",
                                alignItems: "center",
                                padding: "10px",
                            }}
                        >
                            <Typography className={momCls.rectangle} />
                            <Typography>
                                Items with color badge are created manually
                            </Typography>
                        </Box>
                        {index === selectedIdx[index] && (
                            <div
                                style={{ maxWidth: "96%" }}
                                key={Math.random()}
                            >
                                <MaterialTable
                                    columns={populateColumn(title)}
                                    data={doc}
                                    style={{
                                        boxShadow: "none",
                                        width: "104%",
                                        margin: "10px 0 15px 0",
                                    }}
                                    options={{
                                        sorting: false,
                                        filtering: false,
                                        paging: false,
                                        search: false,
                                        showTitle: false,
                                        toolbar: false,
                                        style: { boxShadow: "none" },
                                        headerStyle: {
                                            backgroundColor: "#f2f2f2",
                                        },
                                        rowStyle: (x) => {
                                            if (x.id % 2 === 0) {
                                                return {
                                                    backgroundColor: "#f2f2f2",
                                                    borderLeft: x.manual_add
                                                        ? "2px solid #0061f7"
                                                        : "",
                                                };
                                            }
                                            return {
                                                borderLeft: x.manual_add
                                                    ? "2px solid #0061f7"
                                                    : "",
                                            };
                                        },
                                    }}
                                />
                            </div>
                        )}
                    </List>
                </Collapse>
            </>
        );
    };

    const TeamMember = (props) => {
        const { type, disabled } = props;
        const userExists = (obj, value) =>
            obj.team_members.some((user) => user.email === value);
        return (
            <>
                <FormControl sx={{ m: 1, minWidth: 120 }}>
                    <SelectUI
                        displayEmpty
                        disabled={disabled}
                        open={dropDownOpen[type]}
                        onClose={() => handleDropDownClose(type)}
                        onOpen={() => handleDropDownOpen(type)}
                        value={teamMemberVal[type]}
                        onChange={(e) => handleDropDownChange(e, type)}
                        input={<OutlinedInput />}
                        className={momCls.assignto}
                        MenuProps={{
                            anchorOrigin: {
                                vertical: "bottom",
                                horizontal: "left",
                            },
                            getContentAnchorEl: null,
                        }}
                        renderValue={
                            teamMemberVal[type] &&
                            teamMemberVal[type].length > 0
                                ? undefined
                                : () => (
                                      <Typography
                                          className={momCls.placeHolder}
                                      >
                                          Select name
                                      </Typography>
                                  )
                        }
                        inputProps={{ "aria-label": "Without label" }}
                        IconComponent={() => (
                            <Grid className={momCls.sortOpenIcon}>
                                {!dropDownOpen[type] ? (
                                    <img
                                        src={dropDownIcon}
                                        alt=""
                                        width={"20px"}
                                        height={"20px"}
                                    />
                                ) : (
                                    <img
                                        src={CloseDropDown}
                                        alt=""
                                        width={"20px"}
                                        height={"20px"}
                                    />
                                )}
                            </Grid>
                        )}
                    >
                        {assignTo.map((team, idxVal) => (
                            <MenuItem
                                key={Math.random()}
                                value={team.value}
                                style={{
                                    display: userExists(
                                        team,
                                        localStorage.getItem("email")
                                    )
                                        ? "flex"
                                        : "none",
                                }}
                                classes={{ root: momCls.rootMenuItem }}
                            >
                                {userExists(
                                    team,
                                    localStorage.getItem("email")
                                ) && team.name}
                                {isIconVisibile ? (
                                    <ListItemSecondaryAction>
                                        <IconButton
                                            edge="end"
                                            className={momCls.editBlock}
                                            onClick={(e) =>
                                                EditData(e, type, team)
                                            }
                                        ></IconButton>
                                    </ListItemSecondaryAction>
                                ) : null}
                            </MenuItem>
                        ))}
                    </SelectUI>
                </FormControl>
            </>
        );
    };

    const removeContent = (
        <Box
            style={{
                fontSize: "14px",
                fontFamily: "Lato",
                textAlign: "left",
                color: "#333333",
                padding: "20px",
            }}
        >
            This action cannot be undone. Please make sure you want to remove
            this section from the MoM
        </Box>
    );

    const removeTitle = (
        <Box
            style={{
                fontSize: "24px",
                fontWeight: "medium",
                fontFamily: "Lato",
                textAlign: "left",
                color: "#333333",
            }}
        >
            {popupcontent?.title}
        </Box>
    );

    const addTitle = (
        <Box
            style={{
                fontSize: "20px",
                fontWeight: "bold",
                fontFamily: "Lato",
                textAlign: "left",
                marginLeft: "20px",
            }}
        >
            Add {popupcontent?.title}
        </Box>
    );

    const buttons = (
        <>
            <Box component="div" display="flex" className={momCls.btngroup}>
                <Button
                    variant="text"
                    onClick={closeActionPopup}
                    className={momCls.popupbtn}
                    disabled={isCompleted}
                >
                    Cancel
                </Button>
                <Button
                    variant="contained"
                    color="primary"
                    className={momCls.popupbtn}
                    onClick={addActionsList}
                    disabled={newTranscriptSummary === "" ? true : false}
                >
                    Add
                </Button>
            </Box>
        </>
    );

    const removeButtons = (
        <>
            <Box component="div" display="flex" className={momCls.groupbtn}>
                <Button
                    onClick={closeActionPopup}
                    className={momCls.popupbtn}
                    variant="contained"
                    style={{
                        textTransform: "none",
                        maxWidth: "400px",
                        maxHeight: "40px",
                        minWidth: "175px",
                        minHeight: "40px",
                        fontSize: "16px",
                        fontWeight: "bold",
                        color: "#286ce2",
                        backgroundColor: "#FFFFFF",
                        borderRadius: "8px",
                        border: "2px solid rgb(240, 245, 255)",
                        fontFamily: "Lato",
                    }}
                >
                    Cancel
                </Button>
                <Button
                    variant="contained"
                    component="label"
                    style={{
                        textTransform: "none",
                        maxWidth: "400px",
                        maxHeight: "40px",
                        minWidth: "175px",
                        minHeight: "40px",
                        fontSize: "16px",
                        fontWeight: "bold",
                        color: "#FFFFFF",
                        backgroundColor: "#1665DF",
                        borderRadius: "8px",
                        fontFamily: "Lato",
                    }}
                    onClick={() => deleteActionPopup()}
                >
                    Remove Section
                </Button>
            </Box>
        </>
    );

    const handelUserData = (item, idx) => {
        dispatch({
            type: "SWITCH_TABS",
            payload: {
                title: 1,
                isUserSelection: true,
                userName: item.speaker_id,
                // momStore['map_username'] && momStore['map_username'][item?.speaker_id], //item.speaker_id,
                from: "mom",
            },
        });
    };
    return (
        <>
            <Paper style={{ margin: "26px 0 0 0" }} elevation={0}>
                <Paper className={momCls.paperbottom}>
                    <Typography className={momCls.title}>
                        {"Minutes of Meeting"}
                    </Typography>
                    <Button
                        variant="outlined"
                        className={momCls.buttonCls}
                        disabled={isCompleted}
                        startIcon={<AddCircleOutline />}
                        onClick={openSection}
                    >
                        Add Section
                    </Button>
                    <Menu
                        id="basic-menu"
                        anchorEl={anchorEl}
                        open={sectionopen}
                        onClose={closeSection}
                        MenuListProps={{
                            "aria-labelledby": "basic-button",
                        }}
                        style={{ top: "40px" }}
                        className={momCls.menuWidth}
                    >
                        <Typography
                            className={momCls.labelCls}
                            style={{ float: "none" }}
                        >
                            {""}
                        </Typography>
                        <Box
                            component="div"
                            display="block"
                            className={momCls.listScroll}
                        >
                            {section.map((item, idx) => {
                                let getVal =
                                    getData &&
                                    Object.keys(getData).map(
                                        (ele) => ele === item
                                    );
                                return (
                                    <Box
                                        component="div"
                                        display="flex"
                                        disabled={
                                            getVal && getVal.includes(true)
                                                ? true
                                                : false
                                        }
                                    >
                                        <Checkbox
                                            checked={
                                                getVal && getVal.includes(true)
                                                    ? true
                                                    : checked[idx]
                                            }
                                            inputProps={{
                                                "aria-label": "controlled",
                                            }}
                                            onChange={() =>
                                                seletedSections(item, idx)
                                            }
                                            className={momCls.checkboxcolor}
                                            disabled={
                                                getVal && getVal.includes(true)
                                                    ? true
                                                    : false
                                            }
                                        />
                                        <MenuItem>{item}</MenuItem>
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
                                variant="contained"
                                color="primary"
                                className={momCls.btnMargin}
                                onClick={addSection}
                            >
                                Done
                            </Button>
                        </Box>
                    </Menu>
                </Paper>
                <Grid container spacing={2} className={momCls.projectblock}>
                    <Grid
                        container
                        item
                        xs={4}
                        direction="column"
                        className={momCls.leftsidew}
                    >
                        <Typography className={momCls.labelCls}>
                            Project Name
                        </Typography>
                        <TextField
                            required
                            type="text"
                            margin="0"
                            variant="outlined"
                            size={"small"}
                            // fullWidth
                            value={projectname}
                            disabled={isCompleted}
                            onChange={(e) =>
                                onchangeTextValue(e, "project_name")
                            }
                            className={momCls.projectTextcls}
                            onBlur={(e) =>
                                onBlurField(e, "project_name", projectname)
                            }
                            placeholder="Project Name"
                        />
                    </Grid>
                    <Grid
                        container
                        item
                        xs={3}
                        direction="column"
                        className={momCls.secondrow}
                    >
                        <Typography className={momCls.labelCls}>
                            Meeting Date
                        </Typography>
                        <MMDatePicker
                            handleDateChange={handleDateChange}
                            customIcon={uploadDate ? true : false}
                            value={uploadDate}
                            disabled={isCompleted}
                            style={
                                isCompleted
                                    ? {
                                          PointerEvent: "none",
                                          background: "red",
                                      }
                                    : {
                                          PointerEvent: "none",
                                          background: "red",
                                      }
                            }
                            placeholder="Upload Date"
                            type="meeting_date"
                        />
                    </Grid>
                    <Grid
                        container
                        item
                        xs={3}
                        direction="column"
                        className={momCls.secondrow}
                    >
                        <Typography className={momCls.labelCls}>
                            MoM Date
                        </Typography>
                        <MMDatePicker
                            className={momCls.datepicker}
                            handleDateChange={(e) =>
                                handleDateChange(e, "mom_generation_date")
                            }
                            customIcon={true}
                            width={"auto"}
                            height={"auto"}
                            disabled={isCompleted}
                            value={dateValue}
                            placeholder="MoM Date"
                        />
                    </Grid>
                </Grid>

                <Grid container spacing={2} className={momCls.secondrowM}>
                    <Grid
                        container
                        item
                        xs={12}
                        direction="column"
                        className={momCls.attendeespaperW}
                    >
                        <Typography className={momCls.labelCls}>
                            {"Attendees (" + attendees.length + ")"}
                        </Typography>
                        <Paper className={momCls.attendeespaper}>
                            {attendees.map((item, idx) => (
                                <Typography
                                    className={momCls.tagcls}
                                    onClick={() => handelUserData(item, idx)}
                                >
                                    {momStore &&
                                        momStore["map_username"] &&
                                        momStore["map_username"][
                                            item?.speaker_id
                                        ]}
                                </Typography>
                            ))}
                        </Paper>
                    </Grid>
                </Grid>

                <Grid container spacing={2} className={momCls.secondrowM}>
                    <Grid
                        container
                        item
                        xs={3}
                        direction="column"
                        className={momCls.secondrow}
                    >
                        <Typography className={momCls.labelCls}>
                            Organizer
                        </Typography>
                        <TextField
                            required
                            type="text"
                            margin="0"
                            variant="outlined"
                            size={"small"}
                            fullWidth
                            disabled={isCompleted}
                            value={organizer}
                            placeholder="Organizer Name"
                            onChange={(e) => onchangeOrganiser(e, "organiser")}
                            className={momCls.projectTextcls}
                            onBlur={(e) =>
                                onBlurField(e, "organiser", organizer)
                            }
                        />
                    </Grid>
                    <Grid
                        container
                        item
                        xs={3}
                        direction="column"
                        className={momCls.secondrow}
                    >
                        <Typography className={momCls.labelCls}>
                            Scrum Teamname
                        </Typography>
                        <TeamMember
                            disabled={isCompleted}
                            type="Scrumteamname"
                        />
                    </Grid>
                    <Grid
                        container
                        item
                        xs={3}
                        direction="column"
                        className={momCls.secondrow}
                    >
                        <Typography className={momCls.labelCls}>
                            Location
                        </Typography>

                        <TextField
                            required
                            type="text"
                            margin="0"
                            variant="outlined"
                            size={"small"}
                            fullWidth
                            disabled={isCompleted}
                            value={locationValue}
                            onChange={(e) => onchangeLocation(e, "location")}
                            className={momCls.projectTextcls}
                            onBlur={(e) =>
                                onBlurField(e, "location", locationValue)
                            }
                            placeholder="Select Location"
                        />
                    </Grid>
                    <Grid
                        container
                        item
                        xs={3}
                        direction="column"
                        className={momCls.secondrow}
                    >
                        <Typography className={momCls.labelCls}>
                            Meeting Duration
                        </Typography>
                        <TextField
                            type="text"
                            margin="0"
                            variant="outlined"
                            size={"small"}
                            fullWidth
                            disabled={isCompleted}
                            className={momCls.projectTextcls}
                            onChange={(e) =>
                                onchangeMeetinduration(e, "meeting_duration")
                            }
                            value={meetingDuration}
                            onBlur={(e) =>
                                onBlurField(
                                    e,
                                    "meeting_duration",
                                    meetingDuration
                                )
                            }
                            placeholder="Meeting Duration"
                        />
                    </Grid>
                </Grid>

                <Grid
                    container
                    spacing={2}
                    className={`${momCls.collapseScroll}`}
                >
                    <Grid container item xs={12} direction="column">
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
            </Paper>
            {isOpenPopup && (
                <Modal
                    title={removeItem ? removeTitle : addTitle}
                    content={removeItem ? removeContent : content}
                    actions={removeItem ? removeButtons : buttons}
                    width={removeItem ? "sm" : "md"}
                    open={true}
                    handleClose={handleModalClose}
                    classesNamesDialog={momCls.modalpadding}
                    classeNameTitle={momCls.modalTitleBar}
                    isContent={true}
                />
            )}
        </>
    );
};
export default MomView;
