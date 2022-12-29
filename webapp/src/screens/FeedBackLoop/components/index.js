import React, { useEffect, useState } from "react";
import { getMom, getMeetingMetaData, getPDFReport } from "store/action/mom";
import { getScrumTeamName } from "store/action/scrumTeam";
import {
    Button,
    IconButton,
    Typography,
    Box,
    TextField,
    Grid,
} from "@material-ui/core";
import useStyles from "screens/FeedBackLoop/styles";
import Tab from "screens/FeedBackLoop/components/Tabs";
import PreviewFile from "screens/FeedBackLoop/components/Preview";
import { ArrowBack } from "@mui/icons-material";
import { useDispatch, useSelector } from "react-redux";
import { updateFeedback } from "store/action/mom";
import { changeFileStatus, getPlayBackFilePath } from "store/action/upload";
import Modal from "components/Modal";
import ShareMoM from "./ShareMoM";
import View from "static/images/visibility_black.svg";
import Play from "static/images/system-play.svg";
import { useHistory } from "react-router-dom";
import { isEmpty } from "utils/utility";

const Feedbackloop = () => {
    const dispatch = useDispatch();
    const history = useHistory();
    const { momJson, error, momStore, meetingmetadata, redirection_mask_id } =
        useSelector((state) => state.momReducer);
    const { title } = useSelector((state) => state.tabsReducer);
    const { playbackFileUrl } = useSelector((state) => state.uploadReducer);

    let urlId = window.location.pathname.split("/");
    let Id = urlId[urlId.length - 1];
    const classes = useStyles();
    const [openPreview, setOpenPreview] = useState(false);
    const [fileName, setFileName] = useState("");
    const [fileSize, setFileSize] = useState("");
    const [mediaPath, setMediaPath] = useState("");
    const [fileSource, setFileSource] = useState("");
    const [textValue, setText] = useState("");
    const [titleValue, setTitle] = useState("");
    const [shareMoMModal, setShareMoMModal] = useState(false);
    let MoMStoreData = momStore?.concatenated_view;

    const stopLoader = () => {
        dispatch({
            type: "STOP_LOADER",
            payload: { isLoader: false },
        });
    };

    useEffect(() => {
        if (
            meetingmetadata.meeting_status &&
            meetingmetadata.meeting_status === "Ready for Review"
        ) {
            dispatch(
                changeFileStatus(Id, { status: "User Review In Progress" })
            );
        }
        dispatch(getMeetingMetaData(Id));
        dispatch(getScrumTeamName());
    }, [dispatch, Id, meetingmetadata.meeting_status]);

    useEffect(() => {
        if (momStore && !("map_username" in momStore)) {
            let finalList = momStore?.concatenated_view?.filter(
                (v, i, a) =>
                    a.findIndex((t) => t.speaker_id === v.speaker_id) === i
            );

            let map_username = {};
            finalList = finalList?.map((item) => {
                if (typeof item.speaker_id === "number") {
                    map_username[item.speaker_id] = `User 0${item.speaker_id}`;
                    map_username[
                        `User 0${item.speaker_id}`
                    ] = `User 0${item.speaker_id}`;
                } else {
                    map_username[item.speaker_id] = item.speaker_id;
                }
                return {
                    ...item,
                    value: map_username[item.speaker_id],
                    label: map_username[item.speaker_id],
                };
            });
            const newProperties = {
                AssignTo: finalList,
                map_username: map_username,
            };
            const updatedMoM = { ...momStore, ...newProperties };
            dispatch({
                type: "UPDATE_MOM",
                payload: { ...updatedMoM },
            });
        }
    }, [dispatch, Id, momStore]);

    const prepareData = () => {
        MoMStoreData = MoMStoreData?.map((item, idx) => {
            return {
                ...item,
                bkp_label: { ...item.label },
                bkp_sentiment: { ...item.sentiment },
                bkp_marker: { ...item.marker },
            };
        });
        momStore.concatenated_view = MoMStoreData;
        updateMomStore(momStore);
    };

    useEffect(() => {
        setTitle(momJson.file_name);
        setText(momJson.overview);
        setFileName(momJson.file_name);
        setFileSize(momJson.file_size);
        setMediaPath(momJson.file);
        setFileSource(playbackFileUrl);
        prepareData();
    }, [
        dispatch,
        momJson.overview,
        momJson.file,
        momJson.file_name,
        momJson.file_size,
        playbackFileUrl,
    ]);

    const handlePreviewOpen = () => {
        setOpenPreview(true);
        dispatch(getPlayBackFilePath(mediaPath));
    };
    const handleBack = () => {
        history.push("/upload");
    };

    const handlePreviewClose = (e) => {
        setOpenPreview(false);
    };

    const HandelText = (e) => {
        setText(e.target.value);
    };

    const saveChanges = () => {
        dispatch({
            type: "SAVE_CHANGES",
            payload: { isSaveChanges: true },
        });
    };

    const saveOverView = (e) => {
        if (momJson.overview !== textValue) {
            momJson.overview = textValue;
            updateJson(momJson);
        }
    };

    const saveTitle = (e) => {
        momJson.file_name = titleValue;
        updateJson(momJson);
        saveChanges();
    };

    const updateJson = (momJson) => {
        dispatch(
            updateFeedback(
                redirection_mask_id ||
                    window?.location?.pathname?.split("/")?.splice(-1)[0],
                momJson
            )
        );
    };
    const updateMomStore = (momStore) => {
        dispatch({
            type: "UPDATE_MOM_STORE",
            payload: { momStore: momStore, momJson: momStore }, // momStore: momJson
        });
    };
    const handleTitle = (e) => {
        setTitle(e.target.value);
    };
    if (error === true) {
        stopLoader();
    }

    const handleShareMoM = async () => {
        if (
            meetingmetadata.meeting_status &&
            meetingmetadata.meeting_status === "Completed"
        ) {
            setShareMoMModal(true);
        } else {
            await dispatch(changeFileStatus(Id, { status: "Completed" }));
            setTimeout(() => {
                window.location.reload(false);
            }, 2000);
        }
    };

    const downloadPDF = () => {
        dispatch(getPDFReport(Id, true));
    };

    const previewPDF = () => {
        dispatch(getPDFReport(Id, false));
    };

    const ShareMoMtitle = (
        <Box
            style={{
                color: "#333333",
                fontSize: "20px",
                fontWeight: "bold",
            }}
        >
            Share MoM
        </Box>
    );

    useEffect(() => {
        dispatch(getMom(Id));
    }, [dispatch, momJson.file_name, Id]);

    return (
        <>
            {!isEmpty(momJson) ? (
                <Box component="div" display="block" width="100%">
                    <Grid container spacing={2} className={classes.headecls}>
                        <Grid
                            container
                            item
                            xs={1}
                            direction="column"
                            style={{ maxWidth: "3%" }}
                        >
                            <IconButton
                                className={classes.playIcon}
                                onClick={() => handleBack()}
                            >
                                <ArrowBack fontSize="medium" />
                            </IconButton>{" "}
                        </Grid>
                        <Grid
                            container
                            item
                            xs={6}
                            direction="column"
                            style={{ maxWidth: "43%" }}
                        >
                            <TextField
                                required
                                type="text"
                                margin="normal"
                                variant="outlined"
                                size={"small"}
                                value={titleValue}
                                className={classes.headeTextcls}
                                onChange={handleTitle}
                                InputProps={{
                                    style: { width: "100%" },
                                }}
                                multiline
                                onBlur={saveTitle}
                                disabled
                            />
                        </Grid>
                        <Grid container item xs={3} direction="column">
                            <Button
                                variant="outlined"
                                className={classes.playbtn}
                                onClick={() => handlePreviewOpen()}
                            >
                                <img
                                    src={Play}
                                    alt=""
                                    style={{ padding: "0 6px" }}
                                />
                                Play Recording
                            </Button>
                        </Grid>
                        <Grid
                            container
                            item
                            xs={3}
                            direction="column"
                            className={classes.sharemom}
                        >
                            {
                                <Button
                                    onClick={previewPDF}
                                    className={classes.previewMomBtn}
                                    variant="text"
                                >
                                    <img
                                        src={View}
                                        alt=""
                                        style={{ padding: "0 6px" }}
                                    />
                                    Preview MoM
                                </Button>
                            }
                            {meetingmetadata.meeting_status === "Completed"
                                ? title === 0 && (
                                      <Button
                                          variant="contained"
                                          color="primary"
                                          className={classes.reviewMomBtn}
                                          onClick={(e) => handleShareMoM()}
                                      >
                                          {"Share MoM"}
                                      </Button>
                                  )
                                : title === 0 && (
                                      <Button
                                          variant="contained"
                                          color="primary"
                                          className={classes.reviewMomBtn}
                                          onClick={(e) => handleShareMoM()}
                                      >
                                          {"Submit MoM"}
                                      </Button>
                                  )}
                        </Grid>
                    </Grid>

                    <Box
                        component="div"
                        width="100%"
                        className={classes.overviewBlock}
                    >
                        <Typography className={classes.overViewTitle}>
                            Overview
                        </Typography>
                        <TextField
                            required
                            type="text"
                            variant="outlined"
                            fullWidth
                            value={textValue}
                            className={classes.root}
                            onChange={HandelText}
                            onBlur={saveOverView}
                            multiline
                        />
                    </Box>

                    <Box
                        component="div"
                        display="flex"
                        width="100%"
                        className={classes.tabsView}
                        mt={2}
                    >
                        <Tab />
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
                    {shareMoMModal && (
                        <Modal
                            title={ShareMoMtitle}
                            content={
                                <ShareMoM
                                    downloadPDF={downloadPDF}
                                    previewPDF={previewPDF}
                                />
                            }
                            width={"md"}
                            open={true}
                            handleClose={() => setShareMoMModal(false)}
                            isContent={true}
                            classeNameTitle={classes.modalTitle}
                        />
                    )}
                </Box>
            ) : (
                <>
                    <Grid item xs={12} className={classes.actionCardCss}>
                        <Box
                            style={{
                                fontSize: "20px",
                                margin: "20rem 0rem 0 0rem",
                                fontWeight: "bold",
                            }}
                        >
                            {momJson.status === 500 ? "No Data Found" : ""}
                        </Box>
                    </Grid>
                </>
            )}
        </>
    );
};

export default Feedbackloop;
