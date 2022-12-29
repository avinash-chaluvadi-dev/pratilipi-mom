import { updateObject } from "utils/utility";

const initialState = {
    summary: {},
};
export default function summaryReducer(state = initialState, action = {}) {
    switch (action.type) {
        case "GET_SUMMARY_SUCCESS":
            return updateObject(state, {
                summary: action.response,
            });
        case "GET_SUMMARY_ERROR":
            return updateObject(state, { summary: action.response });
        case "GET_SUMMARY_PROMISEERROR":
            return updateObject(state, { summary: action.response });
        case "JIRA_POST_SUCCESS":
            return updateObject(state, {
                jiraMessage: action.response.message,
            });
        case "JIRA_POST_ERROR":
            return updateObject(state, { jiraMessage: action.response });
        case "JIRA_POST_PROMISEERROR":
            return updateObject(state, { jiraMessage: action.response });
        default:
            return state;
    }
}
