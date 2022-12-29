import React, { useEffect, useState } from "react";
import "./App.css";
import {
    BrowserRouter as Router,
    Switch,
    Route,
    Redirect,
} from "react-router-dom";
import Dashboard from "screens/Dashboard";
import Upload from "screens/Upload";
import FeedBackLoop from "screens/FeedBackLoop";
import ProtectedRoutes from "components/ProtectedRoutes";
import LoaderComponent from "components/Loader";
import ConfigScreen from "screens/Config";
import { Button, createTheme, ThemeProvider } from "@material-ui/core";
import { useDispatch, useSelector } from "react-redux";
import Modal from "components/Modal";
import Login from "screens/Login";
import { errorMessage } from "store/action/types";
import { customErrorMsg } from "utils/utility";
import { logoutChannel } from "components/Sidebar";
import errorIcon from "static/Icons/Error.png";
import Box from "@material-ui/core/Box";

const theme = createTheme({
    palette: {
        primary: {
            main: "#1464DF",
            secondary: "#32c5ff",
            tertiary: "#f0f5ff",
            contrastText: "#ffffff",
        },
        white: { main: "#ffffff", secondary: "#f8fbff", tertiary: "#F6F6F6" },
        grey: { main: "#f6f6f6", secondary: "#ccc", tertiary: "#e8e8e8" },
        success: { main: "#34B53A" },
        error: { main: "#e02020" },
        warning: { main: "#FFB200" },
        dark: { main: "#333" },
        common: {
            muted: "#666666",
        },
    },
    typography: {
        fontSize: 12,
        body1: { fontSize: 12 },
        body2: { fontSize: 10 },
        span: { fontSize: 6 },
    },
});

function App() {
    const dispatch = useDispatch();
    const [errorMsg, setErrorMsg] = useState("");
    const modalActions = (
        <Button
            style={{
                height: "30px",
                width: "100px",
                // backgroundColor: "#FFB3B3",
            }}
            onClick={() =>
                dispatch({
                    type: errorMessage["COMMON_ERROR"],
                    payload: { showError: false, error: "" },
                })
            }
        >
            {" "}
            Close{" "}
        </Button>
    );
    const { isUserAuthenticated } = useSelector(
        (state) => state.userLoginReducer
    );
    const { showError, showErrorMessage } = useSelector(
        (state) => state.errorReducer
    );

    let urlId = window.location.pathname.split("/");
    let Id = urlId[urlId.length - 1];

    const { redirection_mask_id } = useSelector(
        (state) => state.momReducer || { redirection_mask_id: Id }
    );

    const logoutAllTabs = () => {
        logoutChannel.onmessage = () => {
            localStorage.clear();
            window.location.href = `${window.location.origin}/${process.env.PUBLIC_URL}/login`;
            logoutChannel.close();
        };
    };

    const roleDetails = JSON.parse(localStorage.getItem("userRole"));
    const roleDetailsStringify = localStorage.getItem("userRole");
    useEffect(() => {
        const commorErrorMsgString = JSON.stringify(showErrorMessage);
        const parsedCommonErrorMsg = JSON.parse(commorErrorMsgString);
        setErrorMsg(customErrorMsg(parsedCommonErrorMsg));
    }, [showErrorMessage, roleDetailsStringify, errorMsg]);

    useEffect(() => {
        // if logged out in one tab. Logout in other tabs.
        logoutAllTabs();
    }, []);

    const validAuth = () => {
        if (localStorage.getItem("accessToken") === null) {
            return false;
        }
        return true;
    };

    const title = (
        <Box
            style={{
                color: "#333333",
                fontSize: "20px",
                fontWeight: "bold",
            }}
        >
            <img
                src={errorIcon}
                height="20px"
                width="20px"
                style={{ marginBottom: "-2px", marginRight: "10px" }}
                alt="error"
            />
            Error
        </Box>
    );

    return (
        <ThemeProvider theme={theme}>
            <Router basename={process.env.PUBLIC_URL}>
                <div className="App">
                    <LoaderComponent />
                    <Switch>
                        <Route exact path="/login" component={Login} />
                        {validAuth() || isUserAuthenticated ? (
                            <ProtectedRoutes>
                                <Switch>
                                    {roleDetails?.upload && (
                                        <Route
                                            exact
                                            path={`/(|upload)`}
                                            component={Upload}
                                        />
                                    )}
                                    {roleDetails?.dashboard && (
                                        <Route
                                            exact
                                            path="/dashboard"
                                            component={Dashboard}
                                        />
                                    )}
                                    {roleDetails?.mom && (
                                        <Route
                                            path={`/mom/${redirection_mask_id}`}
                                            component={FeedBackLoop}
                                        />
                                    )}
                                    {roleDetails?.configuration && (
                                        <Route
                                            exact
                                            path="/configuration"
                                            component={ConfigScreen}
                                        />
                                    )}
                                    <Redirect to="/upload" />
                                </Switch>
                            </ProtectedRoutes>
                        ) : (
                            <Redirect to="/login" />
                        )}
                    </Switch>
                </div>
            </Router>
            <Modal
                title={title}
                content={errorMsg}
                actions={modalActions}
                open={showError}
                width={"sm"}
                handleClose={() =>
                    dispatch({
                        type: errorMessage["COMMON_ERROR"],
                        payload: { showError: false, error: "" },
                    })
                }
                titleStyle={{
                    borderRadius: "16px",
                    height: "30px",
                    background: "#f7f7f7 padding-box",
                }}
            />
        </ThemeProvider>
    );
}

export default App;
