import React, { useState, useEffect } from "react";
import {
    Box,
    Typography,
    Button,
    TextField,
    Card,
    FormControl,
    ClickAwayListener,
    IconButton,
    ListItemSecondaryAction,
} from "@material-ui/core";
import { loadAssignedValues } from "screens/FeedBackLoop/commonLogic";
import { OutlinedInput } from "@mui/material";
import MenuItemTag from "@material-ui/core/MenuItem";
import { AccountCircle, Edit } from "@mui/icons-material";
import Select from "@material-ui/core/Select";
import { useDispatch, useSelector } from "react-redux";
import customStyles from "screens/FeedBackLoop/components/DetailedView/styles";
import useMoMStyles from "screens/FeedBackLoop/components/MoMView/useStyles";

const TeamMember = (props) => {
    const { type, idx, value, disabled, assignToChange } = props;
    const detailCls = customStyles();
    const momCls = useMoMStyles();
    const dispatch = useDispatch();
    const { momStore } = useSelector((state) => state.momReducer);

    let MoMStoreData = momStore?.concatenated_view;
    const [teamMemberVal, setTeamMember] = useState({});
    const [dropDownOpen, setdropDownOpen] = useState({});
    const [openAddTeamMem, setOpenAddTeamMem] = useState({});
    const [TeamName, setTeamName] = useState({});
    const [isIconVisibile, setIconVisibile] = useState(false);
    const [SubmitTeamData, setSubmitTeamData] = useState({});
    //   const [isEditData, setEditData] = useState(false);
    const [selectedIdx, setSelectedIdx] = useState(null);
    const [oldTeamName, setOldTeamName] = useState();
    const [ActionType, setActionType] = useState("");
    const [selectedValue, setSelectedValue] = useState(value || "");

    const updateJson = (arr) => {
        dispatch({
            type: "UPDATE_MOM_STORE",
            payload: { momStore: arr },
        });
    };

    const handleDropDownChange = (event, type, idx) => {
        setTeamMember({ ...teamMemberVal, [type]: event.target.value });
        setSelectedValue(event.target.value);
        props.compareData();
        assignToChange(idx, event.target.value);
    };

    const handleDrawerClose = (event, type) => {
        setOpenAddTeamMem({ ...openAddTeamMem, [type]: false });
    };

    const handleDropDownClose = (type) => {
        setdropDownOpen({ [type]: false });
        setIconVisibile(false);
    };

    const handleDropDownOpen = (type, idx) => {
        setdropDownOpen({ [type]: true });
        setSelectedIdx(idx);
        setIconVisibile(true);
    };

    const handleAddTeam = (e, type) => {
        setTeamName({ ...TeamName, [type]: e.target.value });
    };

    const CancelTeamBtn = (e, type) => {
        setSubmitTeamData({ ...SubmitTeamData, [type]: false });
        setTeamName({ ...TeamName, [type]: "" });
        setOpenAddTeamMem({ ...openAddTeamMem, [type]: false });
    };

    const AddTeamBtn = (e, type) => {
        if (ActionType === "edit") {
            momStore["map_username"][oldTeamName[type].speaker_id] =
                TeamName[type];
        }
        if (ActionType === "add") {
            let arr = momStore["ExternalParticipants"]
                ? momStore["ExternalParticipants"]
                : [];
            momStore["ExternalParticipants"] = [
                ...arr,
                {
                    value: TeamName[type],
                    label: TeamName[type],
                    speaker_id: Math.random(),
                },
            ];
            updateJson(momStore);
        }

        props.loadAssignToValues();
        setSubmitTeamData({ ...SubmitTeamData, [type]: false });
        setTeamName({ ...TeamName, [type]: "" });
        setOpenAddTeamMem({ ...openAddTeamMem, [type]: false });
    };

    useEffect(() => {
        props.loadAssignToValues();
    }, []);

    const concatedViewStringify = JSON.stringify(momStore.concatenated_view);
    useEffect(() => {
        MoMStoreData[idx].assign_to = selectedValue;
        momStore.concatenated_view = [...MoMStoreData];
        updateJson(momStore);
    }, [selectedValue, concatedViewStringify]);

    const EditData = (e, type, team) => {
        // setEditData(true);
        setOpenAddTeamMem({ ...openAddTeamMem, [type]: true });
        setdropDownOpen({ [type]: false });
        setIconVisibile(false);
        setTeamName({ ...TeamName, [type]: team.value });
        setOldTeamName({ ...TeamName, [type]: team });
        setActionType("edit");
    };

    const addNewMemer = (e, type) => {
        setOpenAddTeamMem({ ...openAddTeamMem, [type]: true });
        setActionType("add");
    };

    return (
        <>
            <FormControl sx={{ m: 1, minWidth: "fit-content" }}>
                <Select
                    displayEmpty
                    disabled={disabled ? true : false}
                    open={selectedIdx === idx && dropDownOpen[type]}
                    onClose={() => handleDropDownClose(type)}
                    onOpen={() => handleDropDownOpen(type, idx)}
                    value={selectedValue} //teamMemberVal[type]}
                    onChange={(e) => handleDropDownChange(e, type, idx)}
                    input={<OutlinedInput />}
                    className={detailCls.assignto}
                    MenuProps={{
                        anchorOrigin: {
                            vertical: "bottom",
                            horizontal: "left",
                        },
                        getContentAnchorEl: null,
                    }}
                    renderValue={
                        // teamMemberVal[type] && teamMemberVal[type].length > 0
                        selectedValue
                            ? undefined
                            : () => (
                                  <Typography className={detailCls.placeHolder}>
                                      Assign to
                                  </Typography>
                              )
                    }
                    inputProps={{ "aria-label": "Without label" }}
                >
                    <MenuItemTag
                        value=""
                        onClick={(event) => {
                            addNewMemer(event, type, idx);
                        }}
                        className={detailCls.addTeam}
                    >
                        + Assign Participant
                    </MenuItemTag>
                    {momStore.AssignTo &&
                        momStore.AssignTo.map((team) => (
                            <MenuItemTag
                                key={Math.random()}
                                value={team.value}
                                className={detailCls.menu}
                            >
                                {/* <AccountCircle className={detailCls.userIcon} /> */}
                                {team.value}
                                {/* {isIconVisibile ? (
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      className={detailCls.editBlock}
                      onClick={(e) => EditData(e, type, team)}
                    >
                      <Edit className={detailCls.editIcon} />
                    </IconButton>
                  </ListItemSecondaryAction>
                ) : null} */}
                            </MenuItemTag>
                        ))}
                </Select>
            </FormControl>

            {
                // openAddTeamMem[type]
                openAddTeamMem[type] && selectedIdx === idx && (
                    <ClickAwayListener
                        onClickAway={(e) => handleDrawerClose(e, type, idx)}
                    >
                        <Card className={detailCls.addTeamPopUp}>
                            <Box padding="10px">
                                <TextField
                                    fullWidth
                                    placeholder="Enter assignee name"
                                    variant="outlined"
                                    size="small"
                                    value={TeamName[type]}
                                    // autoFocus={true}
                                    onChange={(event) => {
                                        handleAddTeam(event, type, idx);
                                    }}
                                    input={<OutlinedInput />}
                                    className={momCls.projectTextcls}
                                />
                            </Box>
                            <Box alignItems="right" marginLeft="140px">
                                <Button
                                    variant="text"
                                    onClick={(e) => CancelTeamBtn(e, type, idx)}
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
                                    onClick={(e) => AddTeamBtn(e, type, idx)}
                                >
                                    Submit
                                </Button>
                            </Box>
                        </Card>
                    </ClickAwayListener>
                )
            }
        </>
    );
};

export default TeamMember;
