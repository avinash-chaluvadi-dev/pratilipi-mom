import React, { useState, useEffect, useCallback } from "react";
import { Box, Button } from "@material-ui/core";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import DetailedView from "screens/FeedBackLoop/components/DetailedView";
import MomView from "screens/FeedBackLoop/components//MoMView";
import { useDispatch, useSelector } from "react-redux";
import Modal from "components/Modal";
import customStyles from "screens/FeedBackLoop/components/MoMView/useStyles";

const TabComp = (props) => {
    const childFunc = React.useRef(null);
    const momCls = customStyles();
    const dispatch = useDispatch();
    const [scrollReset, setScrollReset] = useState(0);
    const { title } = useSelector((state) => state.tabsReducer);
    const [isOpenPopup, setOpenPopup] = useState(false);
    const { isSaveChanges } = useSelector((state) => state.momReducer);
    const [value, setValue] = React.useState(title);
    const { momStore } = useSelector((state) => state.momReducer);

    useEffect(() => {
        setValue(title);
        window.scrollTo(0, 0);
    }, [dispatch, title, value]);

    const handleChange = (event, newValue, fromModal) => {
        if (fromModal === true) {
            switchTab(newValue);
            dispatch({
                type: "SWITCH_TABS",
                payload: {
                    title: newValue,
                },
            });
            setValue(newValue);
            return;
        }
        if (isSaveChanges && (newValue !== 0 || title !== 0)) {
            setOpenPopup(true);
            return;
        } else {
            switchTab(newValue);
            dispatch({
                type: "SWITCH_TABS",
                payload: {
                    title: newValue,
                },
            });
            setValue(newValue);
        }
    };

    const switchTab = (item) => {
        dispatch({
            type: "START_LOADER",
            payload: { isLoader: true },
        });
        dispatch({
            type: "SWITCH_TABS",
            payload: {
                title: item,
                isUserSelection: false,
                userName: "",
                isHilightCard: false,
                highlitedGroup: "",
                from: "",
            },
        });
    };

    const tabDeatilsData = (item) => {
        if (item === 0) {
            return <MomView />;
        } else if (item === 1) {
            return (
                <DetailedView childFunc={childFunc} scrollReset={scrollReset} />
            );
        }
    };

    const handleModalClose = useCallback(
        (event) => {
            dispatch({
                type: "SAVE_CHANGES",
                payload: { isSaveChanges: false },
            });
            handleChange(event, 0, true);
            setOpenPopup(false);
        },
        [dispatch, isOpenPopup]
    );

    const addActionsList = (event) => {
        dispatch({
            type: "SAVE_CHANGES",
            payload: { isSaveChanges: false },
        });
        handleChange(event, 0, true);
        setOpenPopup(false);
    };

    const saveActionsList = () => {
        childFunc.current("bar", momStore.concatenated_view);
        dispatch({
            type: "SAVE_CHANGES",
            payload: { isSaveChanges: false },
        });
        setOpenPopup(false);
    };

    useEffect(() => {
        window.scrollTo(0, 0);
        setScrollReset(!scrollReset);
    }, [handleModalClose]);

    const buttons = (
        <>
            <Box component="div" display="flex" className={momCls.btngroup}>
                <Button
                    onClick={addActionsList}
                    variant="contained"
                    color="primary"
                    style={{
                        textTransform: "none",
                        marginRight: "10px",
                        maxWidth: "400px",
                        maxHeight: "40px",
                        minWidth: "175px",
                        minHeight: "40px",
                        fontSize: "16px",
                        fontWeight: "bold",
                        fontFamily: "Lato",
                        color: "#286ce2",
                        backgroundColor: "#FFFFFF",
                        borderRadius: "8px",
                        border: "2px solid rgb(240, 245, 255)",
                    }}
                >
                    Close
                </Button>
                <Button
                    autoFocus
                    onClick={saveActionsList}
                    variant="contained"
                    color="primary"
                    style={{
                        textTransform: "none",
                        maxWidth: "400px",
                        maxHeight: "40px",
                        minWidth: "175px",
                        minHeight: "40px",
                        fontSize: "16px",
                        color: "#FFFFFF",
                        backgroundColor: "#1665DF",
                        borderRadius: "8px",
                    }}
                >
                    Save Changes
                </Button>
            </Box>
        </>
    );
    return (
        <>
            <Box
                sx={{
                    width: "100%",
                    bgcolor: "background.paper",
                    height: "30px",
                }}
            >
                <Tabs
                    value={value}
                    onChange={handleChange}
                    style={{
                        border: "1px solid #e9e9e9",
                        width: "25%",
                        borderRadius: "8px",
                        display: "flex",
                        alignItems: "center",
                        minHeight: "37px",
                        maxHeight: "37px",
                        margin: "0 1px 10px 1px",
                    }}
                    variant="fullWidth"
                >
                    <Tab
                        label="MoM"
                        key={0}
                        style={{
                            textTransform: "none",
                            background: value === 0 ? "#056aea" : "#ffffff",
                            color: value === 0 ? "#ffffff" : "#0067f3",
                            fontSize: "16px",
                            fontWeight: "bold",
                            fontFamily: "Lato",
                        }}
                    />
                    <Tab
                        label="Detailed View"
                        key={1}
                        style={{
                            textTransform: "none",
                            background: value === 1 ? "#056aea" : "#ffffff",
                            color: value === 1 ? "#ffffff" : "#0067f3",
                            fontSize: "16px",
                            fontWeight: "bold",
                            fontFamily: "Lato",
                        }}
                    />
                </Tabs>
                {tabDeatilsData(value)}
            </Box>
            {isOpenPopup && (
                <Modal
                    title={<h2>Confirmation</h2>}
                    content={
                        "Please click Save Changes to save your modifcations, else click Close to revert your modifications and exit the page."
                    }
                    actions={buttons}
                    width={"sm"}
                    open={true}
                    handleClose={handleModalClose}
                    classNameTop={momCls.modaltop}
                    titleStyle={{
                        borderRadius: "16px",
                        height: "30px",
                        background: "#f7f7f7 padding-box",
                    }}
                />
            )}
        </>
    );
};
export default TabComp;
