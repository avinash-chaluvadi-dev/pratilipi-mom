import React, { useState, useEffect } from "react";
import { Typography, Box } from "@material-ui/core";
import Select from "@material-ui/core/Select";
import Button from "@material-ui/core/Button";
import ButtonGroup from "@material-ui/core/ButtonGroup";
import InputBase from "@material-ui/core/InputBase";
import { withStyles } from "@material-ui/core/styles";
import globalSyles from "styles";
import useStyles from "./screens/Dashboard/styles";
import { useDispatch, useSelector } from "react-redux";

const BootstrapInput = withStyles((theme) => ({
    input: {
        borderRadius: 4,
        position: "relative",
        backgroundColor: theme.palette.background.paper,
        border: "1px solid #ced4da",
        fontSize: 12,
        padding: "7px",
        fontWeight: "bold",
        color: theme.palette.common.muted,
        transition: theme.transitions.create(["border-color", "box-shadow"]),
    },
}))(InputBase);

const CustomToolBar = () => {
    const dispatch = useDispatch();
    const globalClasses = globalSyles();
    const classes = useStyles();
    const [selectedOptionVal, setSelectedOption] = useState("Participants");
    const [selectedBtn, setSelectedBtn] = useState();
    const { selectedIdx, selectedTab } = useSelector(
        (state) => state.insightsReducer
    );

    useEffect(() => {
        setSelectedBtn(1);
    }, []);

    const updateTabProps = (type, idxVal) => {
        setSelectedBtn(idxVal);
        dispatch({
            type: "SWITCH_INSIGHT_TABS",
            payload: { selectedTab: type, selectedIdx: idxVal },
        });
    };

    const handelParticipantsSelection = async (type, idxVal) => {
        setSelectedBtn(idxVal);
        setSelectedOption(type);
        updateTabProps(type, idxVal);
    };

    const handelEscalationsSelection = (type, idxVal) => {
        setSelectedBtn(idxVal);
        setSelectedOption(type);
        updateTabProps(type, idxVal);
    };

    const handelActionsSelection = (type, idxVal) => {
        setSelectedBtn(idxVal);
        setSelectedOption(type);
        updateTabProps(type, idxVal);
    };

    const handelAppreciationsSelection = async (type, idxVal) => {
        setSelectedBtn(idxVal);
        setSelectedOption(type);
        updateTabProps(type, idxVal);
    };

    return (
        <>
            <Box pt={3} pb={3} pl={2} className={globalClasses.rgbaBorder}>
                <Typography
                    variant="h5"
                    align="left"
                    className={globalClasses.bold}
                >
                    Insights
                </Typography>
            </Box>
            <Box display="flex" justifyContent="space-between" mt={2} p={2}>
                <Box display="block" component="div">
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
                                selectedIdx === 0
                                    ? {
                                          background: "#1563DB",
                                          color: "#fff",
                                          padding: "8px 45px",
                                          margin: "-1px",
                                      }
                                    : {
                                          background: "white",
                                          padding: "8px 45px",
                                      }
                            }
                            onClick={() => {
                                // setSelectedBtn(1);
                                handelParticipantsSelection("Summary", 0);
                            }}
                        >
                            Summary
                        </Button>
                        <Button
                            className={`${globalClasses.bold} ${classes.transformvalue}`}
                            style={
                                selectedIdx === 1
                                    ? {
                                          background: "#1563DB",
                                          color: "#fff",
                                          padding: "8px 45px",
                                          margin: "-1px",
                                      }
                                    : {
                                          background: "white",
                                          padding: "8px 45px",
                                      }
                            }
                            onClick={() => {
                                // setSelectedBtn(1);
                                handelParticipantsSelection("Participants", 1);
                            }}
                        >
                            Participants
                        </Button>

                        <Button
                            className={`${globalClasses.bold} ${classes.transformvalue}`}
                            style={
                                selectedIdx === 2
                                    ? {
                                          background: "#1563DB",
                                          color: "#fff",
                                          padding: "8px 45px",
                                          margin: "-1px",
                                      }
                                    : {
                                          background: "white",
                                          padding: "8px 45px",
                                      }
                            }
                            onClick={() => {
                                // setSelectedBtn(2);
                                handelActionsSelection("Actions", 2);
                            }}
                        >
                            Actions
                        </Button>
                        <Button
                            className={`${globalClasses.bold} ${classes.transformvalue}`}
                            style={
                                selectedIdx === 3
                                    ? {
                                          background: "#1563DB",
                                          color: "#fff",
                                          padding: "8px 45px",
                                          margin: "-1px",
                                      }
                                    : {
                                          background: "white",
                                          padding: "8px 45px",
                                      }
                            }
                            onClick={() => {
                                // setSelectedBtn(3);
                                handelEscalationsSelection("Escalations", 3);
                            }}
                        >
                            Escalations
                        </Button>
                        <Button
                            className={`${globalClasses.bold} ${classes.transformvalue}`}
                            style={
                                selectedIdx === 4
                                    ? {
                                          background: "#1563DB",
                                          color: "#fff",
                                          padding: "8px 45px",
                                          margin: "-1px",
                                      }
                                    : {
                                          background: "white",
                                          padding: "8px 45px",
                                      }
                            }
                            onClick={() => {
                                // setSelectedBtn(4);
                                handelAppreciationsSelection(
                                    "Appreciations",
                                    4
                                );
                            }}
                        >
                            Appreciations
                        </Button>
                    </ButtonGroup>
                </Box>

                {selectedIdx === 1 && (
                    <Select
                        native
                        value={""}
                        label="Age"
                        className={classes.formControl}
                        input={<BootstrapInput />}
                    >
                        <option aria-label="None" value="">
                            Select a team
                        </option>
                    </Select>
                )}
            </Box>
        </>
    );
};

export default CustomToolBar;
