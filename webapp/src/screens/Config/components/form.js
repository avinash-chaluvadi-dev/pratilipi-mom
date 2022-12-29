import React, { useState, useEffect } from "react";
import LabeledInput from "components/FormInput";
import { useDispatch } from "react-redux";
import useStyles from "../useStyles";
import { Box, FormControlLabel } from "@material-ui/core";
import Checkbox from "@mui/material/Checkbox";
import TeamMembers from "./teamMembers";
import AddTeamActions from "./formFooter";
import { addTeam, updateTeam } from "store/action/config";
import RemoveIcon from "static/Icons/remove-red.svg";
import { ConstantValue } from "utils/constant";

const HelperText = () => {
    return (
        <Box display="flex">
            <Box
                component="img"
                src={RemoveIcon}
                style={{ marginRight: "10px" }}
            />
            <span>Please enter a valid email </span>
        </Box>
    );
};

const Form = ({ isEditing, currentTeam, isClone, handleCancel }) => {
    const styles = useStyles();
    const [formData, setFormData] = useState({});
    const [inputList, setInputList] = useState([{ name: "", email: "" }]);
    const [jiraDetails, setJiraDetails] = useState({});
    const [validEmail, setValidEmail] = useState({});
    const [showHelperText, setShowHelperText] = useState({
        manager_email: false,
    });
    const [errorStatus, setErrorStatus] = useState(false);
    const dispatch = useDispatch();

    const currentData = isEditing || isClone ? currentTeam : formData;

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData((formData) => ({
            ...formData,
            [name]: value,
        }));
        if (formData.name !== currentTeam.name) setErrorStatus(false);
    };

    const handleJiraChange = (e) => {
        const data = { ...jiraDetails };
        data[e.target.name] = e.target.value;
        setJiraDetails(data);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        let data = formData;
        data["jira_details"] = jiraDetails;
        data["team_members"] = inputList;
        // The condition used to manage same team name while cloning the team
        if (
            currentData.name !== currentTeam.name ||
            (formData.name && formData.name !== currentTeam.name)
        ) {
            setErrorStatus(false);
            if (!isEditing) {
                dispatch(addTeam(data));
            }
        } else {
            setErrorStatus(true);
        }
        // Call the method when editing is enabled
        if (isEditing) {
            setErrorStatus(false);
            dispatch(updateTeam(currentTeam.id, data));
        }
    };

    const validateEmailFormat = (e) => {
        const x = { ...validEmail };
        x[e.target.name] = e.target.value.includes("@");
        setValidEmail(x);
    };

    const handleHelperText = (e) => {
        const x = { ...showHelperText };
        x[e.target.name] = !e.target.value.includes("@");
        setShowHelperText(x);
    };

    useEffect(() => {
        if (currentData?.team_members?.length > 0)
            setInputList(currentData.team_members);
    }, [currentData.team_members]);

    useEffect(() => {
        setJiraDetails(currentData.jira_details);
    }, [
        currentData.jira_details,
        currentData.name,
        formData.name,
        errorStatus,
    ]);

    useEffect(() => {
        setFormData(currentData);

        return () => {
            setFormData({});
        };
    }, []);
    return (
        <form onSubmit={(e) => handleSubmit(e)}>
            <Box className={styles.TeamFormContainer}>
                <Box display="flex">
                    <LabeledInput
                        name="name"
                        placeholder="Scrum Team Name"
                        label="Scrum Team Name*"
                        required
                        value={formData.name}
                        auto
                        defaultValue={currentData.name}
                        onChange={(e) => handleChange(e)}
                        showError={errorStatus}
                        message={ConstantValue.SAME_TEAM_NAME_ERROR_MESSAGE}
                    />
                    <LabeledInput
                        name="dl_email"
                        placeholder="Scrum Distribution list email"
                        label="Scrum Distribution list email*"
                        required
                        value={formData.dl_email}
                        defaultValue={currentData.dl_email}
                        ml={3}
                        type="email"
                        onChange={(e) => handleChange(e)}
                    />
                </Box>
                <Box display="flex" mt={3} alignItems="center">
                    <LabeledInput
                        name="sme_name"
                        placeholder="Scrum Master/SME Name"
                        label="Scrum Master/SME Name*"
                        required
                        value={formData.sme_name}
                        defaultValue={currentData.sme_name}
                        onChange={(e) => handleChange(e)}
                    />
                    <LabeledInput
                        name="sme_email"
                        placeholder="Scrum Master/SME email"
                        label="Scrum Master/SME email*"
                        required
                        ml={3}
                        value={formData.sme_email}
                        type="email"
                        helperText={showHelperText.sme_email && <HelperText />}
                        defaultValue={currentData.sme_email}
                        onChange={(e) => {
                            handleChange(e);
                            validateEmailFormat(e);
                        }}
                        onBlur={(e) => handleHelperText(e)}
                    />
                    <FormControlLabel
                        className={`${styles.CheckboxContainer} ${
                            validEmail.sme_email && styles.CheckboxOpacity
                        }`}
                        control={
                            <Checkbox
                                className={styles.Checkbox}
                                checked={true}
                                name="sme_email_notification"
                                style={{ opacity: "0.3" }}
                            />
                        }
                        label="Include Scrum master/Subject matter email to send all Actions/Escalations emails"
                    />
                </Box>
                <Box display="flex" mt={3}>
                    <LabeledInput
                        name="po_name"
                        placeholder="Product Owner Name"
                        label="Product Owner Name*"
                        value={formData.po_name}
                        defaultValue={currentData.po_name}
                        onChange={(e) => handleChange(e)}
                        required
                    />
                    <LabeledInput
                        name="po_email"
                        placeholder="Product Owner Email"
                        label="Product Owner Email*"
                        required
                        ml={3}
                        value={formData.po_email}
                        defaultValue={currentData.po_email}
                        type="email"
                        helperText={showHelperText.po_email && <HelperText />}
                        onChange={(e) => {
                            handleChange(e);
                            validateEmailFormat(e);
                        }}
                        onBlur={(e) => handleHelperText(e)}
                    />
                    <FormControlLabel
                        className={`${styles.CheckboxContainer} ${
                            validEmail.po_email && styles.CheckboxOpacity
                        }`}
                        name="po_email_notification"
                        control={
                            <Checkbox
                                className={styles.Checkbox}
                                style={{ opacity: "0.3" }}
                                checked={formData.po_email_notification}
                                defaultChecked={
                                    currentData.po_email_notification
                                }
                            />
                        }
                        label="  Include Product Owner email to send all Actions/Escalations emails"
                    />
                </Box>
                <Box display="flex" mt={3}>
                    <LabeledInput
                        placeholder="Manager Name"
                        label="Manager Name*"
                        name="manager_name"
                        value={formData.manager_name}
                        defaultValue={currentData.manager_name}
                        onChange={(e) => handleChange(e)}
                        required
                    />
                    <LabeledInput
                        placeholder="Manager Email"
                        label="Manager Email*"
                        ml={3}
                        name="manager_email"
                        value={formData.manager_email}
                        defaultValue={currentData.manager_email}
                        type="email"
                        helperText={
                            showHelperText.manager_email && <HelperText />
                        }
                        onChange={(e) => {
                            handleChange(e);
                            validateEmailFormat(e);
                        }}
                        onBlur={(e) => handleHelperText(e)}
                        required
                    />
                    <FormControlLabel
                        className={`${styles.CheckboxContainer} ${
                            validEmail.manager_email && styles.CheckboxOpacity
                        }`}
                        name="manager_email_notification"
                        control={
                            <Checkbox
                                className={styles.Checkbox}
                                style={{ opacity: "0.3" }}
                                checked={formData.manager_email_notification}
                                defaultChecked={
                                    currentData.manager_email_notification
                                }
                            />
                        }
                        label="Include Manager email to send all Actions/Escalations emails"
                    />
                </Box>
            </Box>
            <Box display="flex">
                <LabeledInput
                    name="ticket_no"
                    placeholder="Type Something"
                    label="Jira Epic/User Story link for Actions*"
                    value={jiraDetails?.ticket_no}
                    defaultValue={currentData?.jira_details?.ticket_no}
                    onChange={(e) => handleJiraChange(e)}
                    required
                />
            </Box>
            <TeamMembers inputList={inputList} setInputList={setInputList} />
            <AddTeamActions handleCancel={handleCancel} />
        </form>
    );
};

export const FormTitle = ({ isEditing }) => {
    return (
        <Box style={{ fontFamily: "lato", color: " #333;", fontSize: "20px" }}>
            {isEditing ? "Edit" : "Add"} Team
        </Box>
    );
};

export default Form;
