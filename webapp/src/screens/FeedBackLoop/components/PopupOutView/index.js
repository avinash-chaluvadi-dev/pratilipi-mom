import React, { useState, useEffect } from "react";
import useStyles from "screens/FeedBackLoop/styles";
import {
    Box,
    Typography,
    Grid,
    Paper,
    Tooltip,
    TextField,
} from "@material-ui/core";
import ContentEditable from "react-contenteditable";
import { useDispatch, useSelector } from "react-redux";
import AudioPlayerComponent from "screens/FeedBackLoop/components/AudioPlayer";
import ImageGallary from "components/ImageGallary";
import MMDatePicker from "components/MMDatePicker";
import userIcon from "static/images/user.svg";
import ReactTooltip from "react-tooltip";
import DropDown from "components/DropDown";
import customStyles from "screens/FeedBackLoop/components/DetailedView/styles";
import useSumaryStyles from "screens/FeedBackLoop/components/Summary/styles";
import { getFramesFilePath, getPlayBackFilePath } from "store/action/upload";

const PopUpOutView = (props) => {
    const {
        selectedRowData,
        loadAssignToValues,
        handleChangeTemp,
        handleSelectedText,
        onFocus,
        compareData,
        tooltipOpen,
        filterLabelData,
        labelOptions,
        filterEntitiesData,
        entitiesOptions,
        handleChangeDropDownLabel,
        handleDateChange,
        sentimentOptions,
        handleChangeDropDownEntity,
        handleChangeDropDownSentiments,
        summaryTextObjects,
        onSummaryInputchange,
        isImageGallaryOpen,
        TeamMember,
        updateUserName,
        textObjects,
        usernameValueText,
        disabled,
    } = props;

    const summaryClasses = useSumaryStyles();
    const detailCls = customStyles();
    const classes = useStyles();
    const dispatch = useDispatch();
    const { momStore } = useSelector((state) => state.momReducer);
    const { playbackFileUrl, framesFileUrl } = useSelector(
        (state) => state.uploadReducer
    );
    const [usernameValue, setUserName] = useState(
        usernameValueText[selectedRowData?.idx] ||
            (momStore["map_username"] &&
                momStore["map_username"][selectedRowData?.item?.speaker_id])
    );

    const UpdateUserNamePopUp = (e) => {
        setUserName(e.target.value);
    };

    useEffect(() => {
        if (selectedRowData.item.audio_path) {
            dispatch(getPlayBackFilePath(selectedRowData.item.audio_path));
        }
        if (selectedRowData.item.keyframe_labels[0]?.length > 0) {
            dispatch(
                getFramesFilePath(selectedRowData?.item?.keyframe_labels[0])
            );
        }
    }, [
        dispatch,
        selectedRowData.item.audio_path,
        selectedRowData.item.keyframe_labels.length,
    ]);

    return (
        <>
            <Typography
                className={`${classes.userName} ${classes.modalContentBar}`}
            >
                {"Transcript: "}
            </Typography>
            <Grid
                container
                spacing={3}
                xs={12}
                className={classes.actionCardCss}
            >
                <Paper className={`${classes.paperModalPopup}`}>
                    <Box component="div" display="inline">
                        <Box
                            component="div"
                            display="block"
                            className={classes.profileBar}
                        >
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
                                className={summaryClasses.userText}
                                placeholder="User name"
                                InputProps={{
                                    style: { width: "80px" },
                                }}
                                value={usernameValue}
                                disabled={disabled}
                                onChange={(e) => UpdateUserNamePopUp(e)}
                                onBlur={(e) =>
                                    updateUserName(
                                        e,
                                        selectedRowData?.idx,
                                        usernameValue,
                                        selectedRowData?.item?.speaker_id,
                                        "popup"
                                    )
                                }
                            />
                            <Typography className={classes.timer}>
                                {selectedRowData?.item?.start_time}
                                {" - "}
                                {selectedRowData?.item?.end_time}
                            </Typography>
                            <ReactTooltip delayShow={500} />
                            <ReactTooltip delayShow={500} />
                            <AudioPlayerComponent
                                selectedRowData={selectedRowData}
                                audioUrl={playbackFileUrl}
                            />
                            <DropDown
                                // value={selectedOptionEntity && selectedOptionEntity[idx]}
                                disabled={disabled}
                                value={
                                    filterEntitiesData(
                                        selectedRowData?.item.entities,
                                        entitiesOptions
                                    )?.length > 1
                                        ? [
                                              {
                                                  label:
                                                      "Entities" +
                                                      `(${
                                                          filterEntitiesData(
                                                              selectedRowData
                                                                  ?.item
                                                                  .entities,
                                                              entitiesOptions
                                                          )?.length
                                                      })`,
                                                  value:
                                                      "Entities" +
                                                      `(${
                                                          filterEntitiesData(
                                                              selectedRowData
                                                                  ?.item
                                                                  .entities,
                                                              entitiesOptions
                                                          )?.length
                                                      })`,
                                              },
                                              ...filterEntitiesData(
                                                  selectedRowData?.item
                                                      .entities,
                                                  entitiesOptions
                                              ),
                                          ]
                                        : filterEntitiesData(
                                              selectedRowData?.item.entities,
                                              entitiesOptions
                                          )
                                }
                                onChange={(e) =>
                                    handleChangeDropDownEntity(
                                        e,
                                        selectedRowData?.idx
                                    )
                                }
                                options={entitiesOptions}
                                type={"Entity"}
                                placeholder={"Entities"}
                                isNormal={false}
                                isSearchable={false}
                                isMulti={true}
                            />
                        </Box>
                        <ContentEditable
                            html={textObjects[selectedRowData?.idx]}
                            onChange={handleChangeTemp}
                            placeholder="Enter text"
                            disabled={disabled}
                            onSelect={(e) =>
                                handleSelectedText(e, selectedRowData?.idx)
                            }
                            onFocus={() => onFocus(selectedRowData?.idx)}
                            onBlur={() => compareData(selectedRowData?.idx)}
                            style={{
                                height: "46px",
                                overflowY: "auto",
                                width: "100%",
                                textAlign: "left",
                                color: "#949494",
                                fontSize: "12px",
                                fontWeight: "bold",
                                paddingTop: "2px",
                            }}
                        />
                    </Box>
                </Paper>
            </Grid>
            <Typography
                className={`${classes.userName} ${classes.modalContentTitleBar} `}
            >
                {"Transcript Summary: "}
            </Typography>
            <Grid
                container
                spacing={3}
                xs={12}
                className={classes.actionCardCss}
            >
                <Paper className={`${classes.paperModalPopup}`}>
                    <Grid container className={classes.rootGrid} spacing={2}>
                        <Grid
                            item
                            xs={12}
                            p={0}
                            className={classes.profileBarRight}
                        >
                            <Typography className={classes.timerRight}>
                                {selectedRowData?.item?.start_time}
                                {" - "}
                                {selectedRowData?.item?.end_time}
                            </Typography>
                            <Tooltip
                                title="Change Label"
                                placement="top"
                                open={tooltipOpen}
                                arrow
                            >
                                <DropDown
                                    disabled={disabled}
                                    value={
                                        filterLabelData(
                                            selectedRowData?.item?.label,
                                            labelOptions
                                        ).length > 1
                                            ? [
                                                  {
                                                      label:
                                                          "Labels" +
                                                          `(${
                                                              filterLabelData(
                                                                  selectedRowData
                                                                      ?.item
                                                                      ?.label,
                                                                  labelOptions
                                                              ).length
                                                          })`,
                                                      value:
                                                          "Labels" +
                                                          `(${
                                                              filterLabelData(
                                                                  selectedRowData
                                                                      ?.item
                                                                      ?.label,
                                                                  labelOptions
                                                              ).length
                                                          })`,
                                                  },
                                                  ...filterLabelData(
                                                      selectedRowData?.item
                                                          ?.label,
                                                      labelOptions
                                                  ),
                                              ]
                                            : filterLabelData(
                                                  selectedRowData?.item?.label,
                                                  labelOptions
                                              )
                                    }
                                    onChange={(e) =>
                                        handleChangeDropDownLabel(
                                            e,
                                            selectedRowData?.idx
                                        )
                                    }
                                    options={labelOptions}
                                    type={"Label"}
                                    isNormal={false}
                                    isSearchable={false}
                                    placeholder={"Labels"}
                                    from="popup"
                                />
                            </Tooltip>

                            <TeamMember
                                type="Organizer"
                                disabled={disabled}
                                idx={selectedRowData?.idx}
                                value={selectedRowData?.item?.assign_to}
                                compareData={compareData}
                                loadAssignToValues={loadAssignToValues}
                            />
                            {/* <DatePicker
                value={selectedRowData?.item?.date || null}
                handleDateChange={(e) => handleDateChange(e, selectedRowData?.idx)}
                idx={selectedRowData?.idx}
              /> */}
                            <Grid style={{ width: "80px" }}>
                                <MMDatePicker
                                    handleDateChange={(e) =>
                                        handleDateChange(
                                            e,
                                            selectedRowData?.idx
                                        )
                                    }
                                    disabled={disabled}
                                    idx={selectedRowData?.idx}
                                    customIcon={false}
                                    width={"10px"}
                                    height={"10px"}
                                    value={selectedRowData?.item?.date}
                                    placeholder="Date"
                                    type="DetailView"
                                />
                            </Grid>
                            <DropDown
                                disabled={disabled}
                                value={
                                    filterLabelData(
                                        selectedRowData?.item?.sentiment,
                                        sentimentOptions,
                                        "sentiment"
                                    ).length > 1
                                        ? [
                                              {
                                                  label:
                                                      "sentiments" +
                                                      `(${
                                                          filterLabelData(
                                                              selectedRowData
                                                                  ?.item
                                                                  ?.sentiment,
                                                              sentimentOptions,
                                                              "sentiment"
                                                          ).length
                                                      })`,
                                                  value:
                                                      "sentiments" +
                                                      `(${
                                                          filterLabelData(
                                                              selectedRowData
                                                                  ?.item
                                                                  ?.sentiment,
                                                              sentimentOptions,
                                                              "sentiment"
                                                          ).length
                                                      })`,
                                              },
                                              ...filterLabelData(
                                                  selectedRowData?.item
                                                      ?.sentiment,
                                                  sentimentOptions,
                                                  "sentiment"
                                              ),
                                          ]
                                        : filterLabelData(
                                              selectedRowData?.item?.sentiment,
                                              sentimentOptions,
                                              "sentiment"
                                          )
                                }
                                onChange={(e) =>
                                    handleChangeDropDownSentiments(
                                        e,
                                        selectedRowData?.idx
                                    )
                                }
                                options={sentimentOptions}
                                type={"sentimentIcon"}
                                isNormal={false}
                                isSearchable={false}
                                placeholder={"sentiments"}
                                from="popup"
                            />
                        </Grid>

                        <TextField
                            type="text"
                            multiline
                            rows={2}
                            variant="outlined"
                            size={"small"}
                            fullWidth
                            disabled={disabled}
                            placeholder="Enter Text Here"
                            value={summaryTextObjects[selectedRowData?.idx]}
                            className={detailCls.userText}
                            onChange={(e) => {
                                onSummaryInputchange(e, selectedRowData?.idx);
                            }}
                            onFocus={() => onFocus(selectedRowData?.idx)}
                            onBlur={() => compareData(selectedRowData?.idx)}
                            InputProps={{
                                classes: {
                                    input: classes.thaiTextFieldInputProps,
                                },
                            }}
                        />
                    </Grid>
                </Paper>
            </Grid>
            {isImageGallaryOpen && (
                <ImageGallary
                    selectedRowData={selectedRowData}
                    imageUrl={framesFileUrl}
                />
            )}
        </>
    );
};

export default PopUpOutView;
