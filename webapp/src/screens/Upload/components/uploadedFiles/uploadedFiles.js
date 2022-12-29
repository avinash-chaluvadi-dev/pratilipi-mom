import React, { useEffect, useState } from "react";
import { useHistory } from "react-router-dom";
import Table from "components/Table";
import Card from "@material-ui/core/Card";
import PreviewFile from "../modals/previewFile";
import PlayArrowOutlinedIcon from "@material-ui/icons/PlayArrowOutlined";
import Box from "@material-ui/core/Box";
import IconButton from "@material-ui/core/IconButton";
import ReactTooltip from "react-tooltip";
import FiberManualRecordRoundedIcon from "@material-ui/icons/FiberManualRecordRounded";
import globalStyles from "styles";
import { getAllUploadedFiles, getPlayBackFilePath } from "store/action/upload";
import { useSelector, useDispatch } from "react-redux";
import moment from "moment";
import robot from "static/images/robot.png";
import useStyles from "../../useStyles";

const UploadedFiles = () => {
    const [openPreview, setOpenPreview] = useState(false);
    const [fileName, setFileName] = useState("");
    const [fileSize, setFileSize] = useState("");
    const [fileSource, setFileSource] = useState("");
    const [uploadedFilesCount, setUploadedFilesCount] = useState(0);
    let { uploadedFilesUptoDate, uploadedFiles, playbackFileUrl } = useSelector(
        (state) => state.uploadReducer
    );
    let history = useHistory();
    const dispatch = useDispatch();
    const classes = useStyles();

    useEffect(() => {
        dispatch(getAllUploadedFiles());
        // Refresh the component every 15 seconds when mounted
        const interval = setInterval(() => {
            dispatch(getAllUploadedFiles());
        }, 15 * 1000);

        return () => clearInterval(interval);
    }, [uploadedFilesUptoDate, dispatch]);

    useEffect(() => {
        setFileSource(playbackFileUrl);
        if (uploadedFilesUptoDate) setUploadedFilesCount(uploadedFiles.length);
    }, [uploadedFilesUptoDate, uploadedFiles, playbackFileUrl]);

    const globalClasses = globalStyles();

    const handlePreviewOpen = (rowData) => {
        setOpenPreview(true);
        setFileName(rowData.file.substr(rowData.file.lastIndexOf("/") + 1));
        setFileSize(rowData.file_size);
        const filePath = rowData.file;
        const fileUrlArray = filePath.split(":");
        const transcribeUrlData = fileUrlArray.splice(1, 2);
        const combineUrl = transcribeUrlData.join(":");
        const finalUrl = combineUrl.substring(2);
        dispatch(getPlayBackFilePath(finalUrl));
    };
    const handlePreviewClose = () => {
        setOpenPreview(false);
    };

    const tableHeaderStyle = {
        backgroundColor: "#eeeeee",
        color: "#666666",
        fontSize: 14,
        fontWeight: "bold",
    };

    function styleStatus(status) {
        const statusText = status.toLowerCase();
        const getColor = () => {
            if (statusText === "completed") return "#3bb273";
            else if (statusText === "processing") return "#286ce2";
            else if (statusText === "user review in progress") return "#ff4d61";
            else if (statusText === "ready for review") return "#ff6900";
            else if (statusText === "uploaded") return "#1563DB";
            else return "#f2bc35";
        };

        return (
            <div>
                {statusText === "processing" ? (
                    <Box className={globalClasses.flex}>
                        <img
                            src={robot}
                            alt="Upload"
                            data-tip
                            data-for="processing"
                        />{" "}
                        <Box
                            style={{
                                fontSize: "16px",
                                fontWeight: "bold",
                                color: getColor(),
                            }}
                        >
                            {" "}
                            {status}...
                        </Box>
                        <ReactTooltip
                            id="processing"
                            place="top"
                            effect="solid"
                            multiline={true}
                            border
                            textColor="#333333"
                            backgroundColor="#ffffff"
                            borderColor="#286ce2"
                            style={{ width: "10px" }}
                        >
                            Background API process has been started.
                            <br /> Please wait until it is ready for user
                            review.
                        </ReactTooltip>
                    </Box>
                ) : (
                    <Box className={globalClasses.flex}>
                        <FiberManualRecordRoundedIcon
                            style={{
                                fontSize: "16px",
                                fontWeight: "bold",
                                color: getColor(),
                            }}
                        />{" "}
                        <Box
                            style={{
                                fontSize: "16px",
                                fontWeight: "bold",
                                color: getColor(),
                            }}
                        >
                            {" "}
                            {status}
                        </Box>
                    </Box>
                )}
            </div>
        );
    }

    const handleRedirection = (status) => {
        window.sessionStorage.setItem("dontRedirect", false);
        if (status === "Processing") {
            window.sessionStorage.setItem("dontRedirect", true);
        }
    };

    const columns = [
        {
            field: "play",
            title: "",
            cellStyle: {
                width: "1%",
                paddingRight: "0px",
            },
            headerStyle: {
                width: "1%",
                paddingRight: "0px",
            },
            disableClick: true,
            render: (rowData) => (
                <Box>
                    <IconButton
                        onClick={() => handlePreviewOpen(rowData)}
                        aria-label="play"
                        data-tip
                        data-for="play"
                        size="small"
                    >
                        <PlayArrowOutlinedIcon fontSize="small" />
                    </IconButton>
                    <ReactTooltip
                        id="play"
                        place="top"
                        effect="solid"
                        border
                        textColor="#333333"
                        backgroundColor="#ffffff"
                        borderColor="#286ce2"
                    >
                        Play recording
                    </ReactTooltip>
                </Box>
            ),
        },
        {
            field: "name",
            title: "File name",
            cellStyle: {
                width: "42%",
                paddingLeft: "0px",
                borderRight: "1px solid #e5e5e5",
            },
            headerStyle: {
                width: "42%",
                paddingLeft: "0px",
            },
            render: (rowData) => (
                <Box
                    style={{
                        color: "#333333",
                        fontSize: "16px",
                        fontWeight:
                            rowData.status === "Processing" ? "100" : "bold",
                    }}
                    onClick={() => handleRedirection(rowData.status)}
                >
                    {rowData.file.substring(
                        rowData.file.lastIndexOf("/") + 1,
                        rowData.file.lastIndexOf(".")
                    )}
                </Box>
            ),
        },
        {
            field: "teamName",
            title: "Scrum team name",
            cellStyle: {
                width: "14%",
                borderRight: "1px solid #e5e5e5",
            },
            headerStyle: {
                width: "14%",
            },
            render: (rowData) => (
                <Box
                    style={{
                        color: "#333333",
                        fontSize: "16px",
                        fontWeight:
                            rowData.status === "Processing" ? "100" : "bold",
                    }}
                    onClick={() => handleRedirection(rowData.status)}
                >
                    {rowData.full_team_name}
                </Box>
            ),
        },
        {
            field: "type",
            title: "File type",
            cellStyle: {
                width: "10%",
                borderRight: "1px solid #e5e5e5",
            },
            headerStyle: {
                width: "10%",
            },
            render: (rowData) => {
                return (
                    <Box
                        style={{
                            color: "#333333",
                            fontSize: "16px",
                            fontWeight:
                                rowData.status === "Processing"
                                    ? "100"
                                    : "bold",
                        }}
                        onClick={() => handleRedirection(rowData.status)}
                    >
                        {rowData.file.substr(rowData.file.lastIndexOf("."))}
                    </Box>
                );
            },
        },
        {
            field: "date",
            title: "Date",
            cellStyle: {
                width: "10%",
                borderRight: "1px solid #e5e5e5",
            },
            headerStyle: {
                width: "10%",
            },
            render: (rowData) => {
                let date = moment(rowData.date, moment.defaultFormat).format(
                    "DD MMM YYYY hh:mm a"
                );
                return (
                    <Box
                        style={{
                            color: "#333333",
                            fontSize: "16px",
                            fontWeight:
                                rowData.status === "Processing"
                                    ? "100"
                                    : "bold",
                        }}
                        onClick={() => handleRedirection(rowData.status)}
                    >
                        {date}
                    </Box>
                );
            },
        },
        {
            field: "status",
            title: "Status",
            cellStyle: {
                width: "16%",
                borderRight: "1px solid #e5e5e5",
            },
            headerStyle: {
                width: "16%",
            },
            render: (rowData) => (
                <Box onClick={() => handleRedirection(rowData.status)}>
                    {styleStatus(rowData.status)}
                </Box>
            ),
        },
    ];

    const redirectFun = (id, selectedRow) => {
        if (window.sessionStorage.getItem("dontRedirect") !== "true") {
            dispatch({
                type: "SAVE_CHANGES",
                payload: { isSaveChanges: false },
            });
            dispatch({
                type: "UPDATE_MOM_APIS",
                payload: {
                    momJsonUptoDate: false,
                    momJson: {},
                    redirection_mask_id: selectedRow.masked_request_id,
                },
            });
            dispatch({
                type: "SWITCH_TABS",
                payload: {
                    title: 0,
                },
            });
            history.push(`/mom/${selectedRow.masked_request_id}`);
        }
    };

    return (
        <Card className={classes.uploadFiles}>
            <Box
                style={{
                    fontSize: "20px",
                    fontWeight: "bold",
                    textAlign: "left",
                    color: "#464546",
                    lineHeight: 3,
                }}
            >
                My Uploads ({uploadedFilesCount})
                <Table
                    data={uploadedFiles}
                    redirectFun={redirectFun}
                    columns={columns}
                    hideToolbar={true}
                    dense={true}
                    headerStyle={tableHeaderStyle}
                    disablePaging={false}
                    longTable={true}
                />
            </Box>
            {openPreview && (
                <PreviewFile
                    openPreview={true}
                    handlePreviewClose={handlePreviewClose}
                    fileName={fileName}
                    fileSize={fileSize}
                    src={fileSource}
                />
            )}
        </Card>
    );
};

export default UploadedFiles;
