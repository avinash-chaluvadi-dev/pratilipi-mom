import { Summary } from "./types";
import service from "api/service";

export const getSummary = (params) => (dispatch) => {
    service.getWithParams(
        dispatch,
        "/api/participants/",
        params,
        Summary["GET_SUMMARY"]
    );
};
export const jiraPost = (body) => (dispatch) => {
    service.post(dispatch, "/api/create-issue/", body, Summary["JIRA_POST"]);
};
