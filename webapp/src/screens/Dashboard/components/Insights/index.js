import React, { useState, useEffect } from "react";
import { Box, Button, Card } from "@material-ui/core";
import Insight from "./insights";
import { useDispatch, useSelector } from "react-redux";
import useStyles from "screens/Dashboard/styles";
import Modal from "components/Modal";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import MaterialTable from "material-table";
import outlookIcon from "static/Icons/outlook.png";
import JiraIcon from "static/Icons/Jira.svg";
import moment from "moment";
import { jiraPost } from "store/action/summary";

const Insights = () => {
    const classes = useStyles();
    const [isHover, setIsHover] = useState({});
    const [setIsHoverType] = useState("");
    const [isJiraAllCheckbox, setIsJiraAllCheckbox] = useState({});
    const [isOutlookAllCheckbox, setIsOutlookAllCheckbox] = useState({});
    const [selectedCard, setSelectedCard] = useState({});
    const [isOpenConfirmPopup, setIsOpenConfirmPopup] = useState(false);
    const { summary } = useSelector((state) => state.summaryReducer);
    const [summaryStat, setSummaryStat] = useState({});
    const [teamArray, setTeamArray] = useState([]);
    const [jiraArr, setJiraArr] = useState({});
    const [outlookArr, setOutlookArr] = useState({});

    useEffect(() => {
        if (summary) {
            setSummaryStat({ ...summary });
        }
    }, [summary]);

    useEffect(() => {
        if (summaryStat?.team_info)
            setTeamArray(Object.keys(summaryStat?.team_info));
    }, [summaryStat?.team_info]);

    const dispatch = useDispatch();
    const tableColumns = (item, idx) => {
        return [
            {
                title: "S.NO",
                field: "id",
                render: (rowData) => {
                    return <Box>{rowData.tableData.id + 1}</Box>;
                },
                cellStyle: {
                    borderRight: "1px solid #eee",
                    textAlign: "left",
                    fontSize: "16px",
                    width: "4%",
                    borderBottom: "0px",
                    padding: "5px 20px",
                    color: "#333",
                    fontFamily: "Lato",
                },
                headerStyle: {
                    borderRight: "1px solid #eee",
                    textAlign: "left",
                    color: "#666",
                    width: "4%",
                    fontFamily: "Lato",
                    fontSize: "14px",
                    fontWeight: "bold",
                    borderBottom: "0px",
                    padding: "5px 20px",
                },
            },
            {
                title: "Discussed items",
                field: "transcript",
                render: (rowData) => {
                    return (
                        <Box
                            style={{
                                wordBreak: "break-word",
                            }}
                        >
                            {rowData.transcript}
                        </Box>
                    );
                },
                cellStyle: {
                    borderRight: "1px solid #eee",
                    fontSize: "16px",
                    cursor: "pointer",
                    width: "50%",
                    borderBottom: "0px",
                    padding: "5px 20px",
                    color: "#333",
                    fontFamily: "Lato",
                },
                headerStyle: {
                    borderRight: "1px solid #eee",
                    textAlign: "left",
                    color: "#666",
                    width: "50%",
                    fontFamily: "Lato",
                    fontSize: "14px",
                    fontWeight: "bold",
                    cursor: "pointer",
                    borderBottom: "0px",
                    padding: "5px 20px",
                },
            },
            {
                title: "Date",
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
                    borderRight: "1px solid #eee",
                    fontSize: "16px",
                    cursor: "pointer",
                    width: "20%",
                    borderBottom: "0px",
                    padding: "5px 20px",
                    color: "#333",
                    fontFamily: "Lato",
                },
                headerStyle: {
                    borderRight: "1px solid #eee",
                    textAlign: "left",
                    color: "#666",
                    width: "20%",
                    fontFamily: "Lato",
                    fontSize: "14px",
                    fontWeight: "bold",
                    cursor: "pointer",
                    borderBottom: "0px",
                    padding: "5px 20px",
                },
            },
            {
                title: "Owner",
                field: "owner",
                type: "numeric",
                cellStyle: {
                    borderRight: "1px solid #eee",
                    textAlign: "left",
                    fontSize: "16px",
                    width: "20%",
                    borderBottom: "0px",
                    padding: "5px 20px",
                    color: "#333",
                    fontFamily: "Lato",
                },
                headerStyle: {
                    borderRight: "1px solid #eee",
                    textAlign: "left",
                    color: "#666",
                    width: "20%",
                    fontFamily: "Lato",
                    fontSize: "14px",
                    fontWeight: "bold",
                    borderBottom: "0px",
                    padding: "5px 20px",
                },
            },
            {
                title: (
                    <div
                        onMouseEnter={() => hoverOn("OutLook")}
                        onMouseLeave={() => hoverOff("OutLook")}
                    >
                        {(isHover["OutLook"] && !isOutlookAllCheckbox[idx]) ||
                        isOutlookAllCheckbox[idx] ? (
                            <input
                                type="checkbox"
                                checked={isOutlookAllCheckbox[idx]}
                                className={classes.checkbox}
                                onChange={(e) => {
                                    onChangeCheckBox(
                                        item,
                                        "header",
                                        e.target.checked,
                                        "OutLook"
                                    );
                                }}
                            />
                        ) : (
                            <img
                                src={outlookIcon}
                                alt=""
                                width="23px"
                                height="23px"
                            />
                        )}
                    </div>
                ),
                field: "OutLook",
                render: (rowdata) => (
                    <input
                        type="checkbox"
                        checked={isOutlookAllCheckbox[idx]}
                        className={classes.checkbox}
                        onChange={(e) => {
                            onChangeCheckBox(
                                item,
                                "body",
                                e.target.checked,
                                "OutLook",
                                rowdata.transcript
                            );
                        }}
                    />
                ),
                cellStyle: {
                    borderRight: "1px solid #eee",
                    color: "primary",
                    fontSize: "16px",
                    cursor: "pointer",
                    width: "3%",
                    borderBottom: "0px",
                    padding: "5px 20px",
                },
                headerStyle: {
                    borderRight: "1px solid #eee",
                    textAlign: "left",
                    color: "#666",
                    width: "3%",
                    fontFamily: "Lato",
                    fontSize: "14px",
                    fontWeight: "bold",
                    cursor: "pointer",
                    borderBottom: "0px",
                    padding: "5px 20px",
                },
            },
            {
                title: (
                    <div
                        onMouseEnter={() => hoverOn("Jira")}
                        onMouseLeave={() => hoverOff("Jira")}
                    >
                        {(isHover["Jira"] && !isJiraAllCheckbox[idx]) ||
                        isJiraAllCheckbox[idx] ? (
                            <input
                                type="checkbox"
                                checked={isJiraAllCheckbox[idx]}
                                className={classes.checkbox}
                                onChange={(e) => {
                                    onChangeCheckBox(
                                        item,
                                        "header",
                                        e.target.checked,
                                        "Jira"
                                    );
                                }}
                            />
                        ) : (
                            <img
                                src={JiraIcon}
                                alt=""
                                width="23px"
                                height="23px"
                            />
                        )}
                    </div>
                ),
                field: "Jira",
                render: (rowdata) => (
                    <input
                        type="checkbox"
                        checked={isJiraAllCheckbox[idx]}
                        className={classes.checkbox}
                        onChange={(e) => {
                            onChangeCheckBox(
                                item,
                                "body",
                                e.target.checked,
                                "Jira",
                                rowdata.transcript
                            );
                        }}
                    />
                ),
                cellStyle: {
                    borderRight: "1px solid #eee",
                    color: "primary",
                    fontSize: "16px",
                    cursor: "pointer",
                    width: "3%",
                    borderBottom: "0px",
                    padding: "5px 20px",
                },
                headerStyle: {
                    borderRight: "1px solid #eee",
                    textAlign: "left",
                    color: "#666",
                    width: "3%",
                    fontFamily: "Lato",
                    fontSize: "14px",
                    fontWeight: "bold",
                    cursor: "pointer",
                    borderBottom: "0px",
                    padding: "5px 20px",
                },
            },
        ];
    };

    const {
        selectedCardInfo: { cardTitle, openCardModal },
    } = useSelector((state) => state.dashboardActionTabReducer);

    const hoverOn = (type) => {
        setIsHover({ [type]: true });
        setIsHoverType(type);
    };

    const hoverOff = (type) => {
        setIsHover({ [type]: false });
        setIsHoverType("");
    };

    const hoverOnHeader = (item, idx) => {
        setSelectedCard(idx);
    };

    const hoverOffHeader = (item, idx) => {
        setSelectedCard("");
    };

    const onChangeCheckBox = (team, from, value, type, transcript) => {
        if (from === "header" && type === "OutLook") {
            if (value) {
                isOutlookAllCheckbox[selectedCard] = value;
                outlookArr[team] = [];
                var checkeddata = populateData(team);
                for (let row of checkeddata) {
                    outlookArr[team].push(row.transcript);
                }
            } else {
                delete isOutlookAllCheckbox[selectedCard];
                delete outlookArr[team];
            }
        } else if (from === "body" && type === "OutLook") {
            if (team in outlookArr) {
                if (value) {
                    outlookArr[team].push(transcript);
                } else {
                    outlookArr[team] = removeSelectedTranscript(
                        outlookArr[team],
                        transcript
                    );
                    if (outlookArr[team].length === 0) {
                        delete outlookArr[team];
                    }
                }
            } else {
                outlookArr[team] = [transcript];
            }
        } else if (from === "header" && type === "Jira") {
            if (value) {
                isJiraAllCheckbox[selectedCard] = value;
                jiraArr[team] = [];
                checkeddata = populateData(team);
                for (let row of checkeddata) {
                    jiraArr[team].push(row.transcript);
                }
            } else {
                delete isJiraAllCheckbox[selectedCard];
                delete jiraArr[team];
            }
        } else if (from === "body" && type === "Jira") {
            if (team in jiraArr) {
                if (value) {
                    jiraArr[team].push(transcript);
                } else {
                    jiraArr[team] = removeSelectedTranscript(
                        jiraArr[team],
                        transcript
                    );
                    if (jiraArr[team].length === 0) {
                        delete jiraArr[team];
                    }
                }
            } else {
                jiraArr[team] = [transcript];
            }
        }
    };

    function removeSelectedTranscript(arr, value) {
        var index = arr.indexOf(value);
        if (index > -1) {
            arr.splice(index, 1);
        }
        return arr;
    }

    const handleModalClose = () => {
        dispatch({
            type: "SELECTED_ACTION_ROW_DATA",
            payload: {
                selectedCardInfo: { cardTitle: "", openCardModal: false },
            },
        });
        setIsOutlookAllCheckbox({});
        setIsJiraAllCheckbox({});
        setJiraArr({});
        setOutlookArr({});
    };

    const populateCount = (teamName) => {
        let labelEntries = summaryStat.team_info[teamName];
        if (cardTitle === "Actions")
            return labelEntries.Action?.transcripts?.length || 0;
        else if (cardTitle === "Escalations")
            return labelEntries.Escalation?.transcripts?.length || 0;
        else if (cardTitle === "Appreciations")
            return labelEntries.Appreciation?.transcripts?.length || 0;
    };

    const populateData = (teamName) => {
        let labelEntries = summaryStat.team_info[teamName];
        if (cardTitle === "Actions") return labelEntries.Action?.transcripts;
        else if (cardTitle === "Escalations")
            return labelEntries.Escalation?.transcripts;
        else if (cardTitle === "Appreciations")
            return labelEntries.Appreciation?.transcripts;
    };

    const tableBody = (
        <div className={classes.Accordianroot}>
            {teamArray.map((item, idx) => (
                <Accordion
                    elevation={0}
                    onMouseEnter={() => hoverOnHeader(item, idx)}
                    onMouseLeave={() => hoverOffHeader(item, idx)}
                >
                    <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls="panel1a-content"
                        id="panel1a-header"
                        className={classes.MuiAccordionSummary}
                    >
                        <Typography className={classes.Accordianheading}>
                            {item} (
                            {populateCount(item) ? populateCount(item) : "0"})
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <MaterialTable
                            columns={tableColumns(item, idx)}
                            data={populateData(item)}
                            style={{
                                boxShadow: "none",
                                margin: "10px 0 1px 0",
                                borderRadius: "8px",
                                border: "4px solid #f0f0f0",
                            }}
                            options={{
                                sorting: false,
                                filtering: false,
                                paging: false,
                                search: false,
                                showTitle: false,
                                toolbar: false,
                                style: { boxShadow: "none" },
                                headerStyle: { backgroundColor: "#f2f2f2" },
                                rowStyle: (x) => {
                                    if (x.tableData.id % 2 === 1) {
                                        return {
                                            backgroundColor: "#f2f2f2",
                                        };
                                    }
                                },
                            }}
                        />
                    </AccordionDetails>
                </Accordion>
            ))}
        </div>
    );

    const openConfirmationModal = () => {
        setIsOpenConfirmPopup(true);
    };

    const handleConfirmClose = () => {
        setIsOpenConfirmPopup(!isOpenConfirmPopup);
    };

    const submitJiraPost = () => {
        dispatch(jiraPost(jiraArr));
        setIsOpenConfirmPopup(!isOpenConfirmPopup);
    };

    const buttons = () => {
        let enable =
            Object.keys(jiraArr).length !== 0 ||
            Object.keys(outlookArr).length !== 0;
        return (
            <Button
                variant="contained"
                color="primary"
                className={classes.popupSubmitBtn}
                style={{
                    textTransform: "none",
                    right: 0,
                    opacity: enable ? 1 : 0.5,
                    pointerEvents: enable ? "auto" : "none",
                    fontSize: "16px",
                    fontWeight: "bold",
                    fontFamily: "Lato",
                    color: "#FFFFFF",
                    backgroundColor: "#1665DF",
                    borderRadius: "8px",
                }}
                onClick={() => openConfirmationModal()}
            >
                Submit
            </Button>
        );
    };

    const confirmModalBtns = (
        <>
            <Button
                variant="contained"
                color="primary"
                className={classes.popupSubmitBtn}
                style={{
                    textTransform: "none",
                    fontSize: "16px",
                    fontWeight: "bold",
                    fontFamily: "Lato",
                    color: "#286ce2",
                    backgroundColor: "#ffffff",
                    borderRadius: "8px",
                    border: "2px solid #f0f5ff",
                }}
                onClick={handleConfirmClose}
            >
                Cancel
            </Button>
            <Button
                variant="contained"
                color="primary"
                className={classes.popupSubmitBtn}
                style={{
                    textTransform: "none",
                    fontSize: "16px",
                    fontWeight: "bold",
                    fontFamily: "Lato",
                    color: "#FFFFFF",
                    backgroundColor: "#1665DF",
                    borderRadius: "8px",
                }}
                onClick={submitJiraPost}
            >
                Submit
            </Button>
        </>
    );

    const title = (
        <Box
            style={{
                color: "#333333",
                fontSize: "20px",
                fontWeight: "bold",
            }}
        >
            {cardTitle}
        </Box>
    );
    const jiraConfirmationtitle = (
        <Box
            style={{
                color: "#333333",
                fontSize: "20px",
                fontWeight: "bold",
            }}
        >
            Are you sure you want to submit ?
        </Box>
    );
    const jiraconfirmationtitleContent = (
        <Box>
            Email will be triggered to registered email ids and tasks creation
            will be done under the provided Epic link. Go to configuration
            screen to change these details.
        </Box>
    );

    return (
        <>
            <Box mt={4}>
                <Card
                    style={{ height: "69vh", overflow: "auto" }}
                    className={classes.mainDashboardCard}
                    alignItems="center"
                    justify="center"
                >
                    <Insight />
                </Card>
            </Box>
            {openCardModal && (
                <Modal
                    title={title}
                    content={tableBody}
                    actions={buttons()}
                    width={"md"}
                    open={true}
                    handleClose={handleModalClose}
                    classesNamesDialog={classes.modalWidtHeight}
                />
            )}
            {isOpenConfirmPopup && (
                <Modal
                    title={jiraConfirmationtitle}
                    content={jiraconfirmationtitleContent}
                    actions={confirmModalBtns}
                    width={"sm"}
                    open={true}
                    handleClose={handleConfirmClose}
                    classeNameTitle={classes.modalTitle}
                />
            )}
        </>
    );
};

export default Insights;
