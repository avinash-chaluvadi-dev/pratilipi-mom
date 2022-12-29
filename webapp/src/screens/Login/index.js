import React, { useState, useEffect } from "react";
import Box from "@material-ui/core/Box";
import Grid from "@material-ui/core/Grid";
import Divider from "@material-ui/core/Divider";
import Typography from "@material-ui/core/Typography";
import Button from "@material-ui/core/Button";
import FormHelperText from "@mui/material/FormHelperText";
import pratilipiLogo from "static/images/pratilipi_logo.svg";
import momLogo from "static/images/mom_logo.png";
import { useSelector, useDispatch } from "react-redux";
import { userLogin } from "store/action/login";
import useStyles from "./styles";
import { useHistory } from "react-router-dom";
import OutlinedInput from "@mui/material/OutlinedInput";
import InputAdornment from "@mui/material/InputAdornment";
import FormControl from "@mui/material/FormControl";
import Visibility from "static/images/iconography-system-visible.svg";
import VisibilityOff from "static/images/iconography-system-invisible.svg";
import ClearIcon from "@mui/icons-material/Clear";
import IconButton from "@mui/material/IconButton";

const Login = () => {
    const loginReducer = useSelector((state) => state.userLoginReducer);
    const [id, setId] = useState("");
    const [password, setPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);
    const loginCls = useStyles();
    const dispatch = useDispatch();
    const history = useHistory();
    const [invalidUser, setInvalidUser] = useState(false);
    const [invalidPassword, setInvalidPassword] = useState(false);

    const onFormSubmit = (e) => {
        e.preventDefault();
        let formData = {
            email: id,
            password: password,
        };
        if (formData.email === "" || formData.password === "") {
            setInvalidPassword(true);
            setInvalidUser(true);
        } else {
            dispatch(userLogin(formData));
        }
    };

    useEffect(() => {
        const SUCCESSSTATUS = [200, 201, 204];
        if (localStorage.getItem("accessToken") !== null) {
            setInvalidPassword(false);
            setInvalidUser(false);
            const role = loginReducer.role || {};
            history.push(`/${role.home_screen}`);
        }

        if (
            loginReducer &&
            loginReducer.authenticationMessage &&
            !SUCCESSSTATUS.includes(loginReducer.authenticationMessage.status)
        ) {
            setInvalidUser(true);
            setInvalidPassword(true);
        }

        if (id === "" && password === "") {
            setInvalidPassword(false);
            setInvalidUser(false);
        }
    }, [loginReducer, history, id, password]);

    return (
        <Grid container direction="column" className={loginCls.outerGrid}>
            <Grid item lg={10} md={20} xs={10}>
                <Box className={loginCls.outerBox}>
                    <Box className={loginCls.logoBox}>
                        <img src={pratilipiLogo} alt="Upload" />
                        <img
                            src={momLogo}
                            className={loginCls.momLogo}
                            alt="mom"
                        />
                        <Typography className={loginCls.momSummary}>
                            Automatically convert your meeting recordings into
                            Minutes of Meeting using our Intelligent Speech
                            Transcription and Analytics platform.
                        </Typography>
                    </Box>
                    <Divider
                        className={loginCls.divider}
                        orientation="vertical"
                    />
                    <Box
                        ml={10}
                        component="form"
                        autoComplete="off"
                        onSubmit={onFormSubmit}
                    >
                        <FormControl className={loginCls.loginForm}>
                            <Typography className={loginCls.headerText}>
                                Log In
                            </Typography>
                            <Typography>&nbsp;</Typography>
                            <Typography className={loginCls.bodyText}>
                                Email/Domain Id*
                            </Typography>
                            {invalidUser ? (
                                <FormControl error variant="standard">
                                    <OutlinedInput
                                        size="small"
                                        className={loginCls.loginForm}
                                        placeholder="Enter email/domain id"
                                        value={id}
                                        onChange={(e) => setId(e.target.value)}
                                    />
                                    <FormHelperText
                                        className={loginCls.errorText}
                                    >
                                        <ClearIcon />
                                        Please enter a valid email/domain id
                                    </FormHelperText>
                                </FormControl>
                            ) : (
                                <FormControl variant="standard">
                                    <OutlinedInput
                                        size="small"
                                        className={loginCls.loginForm}
                                        placeholder="Enter email/domain id"
                                        value={id}
                                        onChange={(e) => setId(e.target.value)}
                                    />
                                </FormControl>
                            )}
                            <Typography>&nbsp;</Typography>
                            <Typography className={loginCls.bodyText}>
                                Password
                            </Typography>
                            {invalidPassword ? (
                                <FormControl error variant="standard">
                                    <OutlinedInput
                                        size="small"
                                        className={loginCls.loginForm}
                                        helperText="Incorrect entry."
                                        type={
                                            showPassword ? "text" : "password"
                                        }
                                        placeholder="Enter password"
                                        value={password}
                                        onChange={(e) =>
                                            setPassword(e.target.value)
                                        }
                                        endAdornment={
                                            <InputAdornment position="end">
                                                <IconButton
                                                    onClick={(e) =>
                                                        setShowPassword(
                                                            !showPassword
                                                        )
                                                    }
                                                    onMouseDown={(event) =>
                                                        event.preventDefault()
                                                    }
                                                    edge="end"
                                                >
                                                    {showPassword ? (
                                                        <img
                                                            src={Visibility}
                                                            alt="Visbility off"
                                                        />
                                                    ) : (
                                                        <img
                                                            src={VisibilityOff}
                                                            alt="Visbility on"
                                                        />
                                                    )}
                                                </IconButton>
                                            </InputAdornment>
                                        }
                                    />
                                    <FormHelperText
                                        className={loginCls.errorText}
                                    >
                                        <ClearIcon />
                                        Please enter a valid password
                                    </FormHelperText>
                                </FormControl>
                            ) : (
                                <FormControl>
                                    <OutlinedInput
                                        size="small"
                                        className={loginCls.loginForm}
                                        helperText="Incorrect entry."
                                        type={
                                            showPassword ? "text" : "password"
                                        }
                                        placeholder="Enter password"
                                        value={password}
                                        onChange={(e) =>
                                            setPassword(e.target.value)
                                        }
                                        endAdornment={
                                            <InputAdornment position="end">
                                                <IconButton
                                                    onClick={(e) =>
                                                        setShowPassword(
                                                            !showPassword
                                                        )
                                                    }
                                                    onMouseDown={(event) =>
                                                        event.preventDefault()
                                                    }
                                                    edge="end"
                                                >
                                                    {showPassword ? (
                                                        <img
                                                            src={Visibility}
                                                            alt="Visbility off"
                                                        />
                                                    ) : (
                                                        <img
                                                            src={VisibilityOff}
                                                            alt="Visbility on"
                                                        />
                                                    )}
                                                </IconButton>
                                            </InputAdornment>
                                        }
                                    />
                                </FormControl>
                            )}
                        </FormControl>
                        <Typography>&nbsp;</Typography>
                        <Typography>
                            <Button
                                variant="contained"
                                color="primary"
                                size="large"
                                type="submit"
                                className={loginCls.button}
                            >
                                Log In
                            </Button>
                        </Typography>
                    </Box>
                </Box>
            </Grid>
        </Grid>
    );
};

export default Login;
