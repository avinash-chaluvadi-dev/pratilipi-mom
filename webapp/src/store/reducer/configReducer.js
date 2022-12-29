import { updateObject } from "utils/utility";

const initialState = {
    teams: [],
    drafts: [],
    teamsUptoDate: false,
    closeModal: false,
};

export default function confgiReducer(state = initialState, action = {}) {
    switch (action.type) {
        case "GET_ALL_TEAMS_SUCCESS":
            return updateObject(state, {
                teams: action.response,
                teamsUptoDate: true,
            });
        case "GET_ALL_TEAMS_ERROR":
            return updateObject(state, {
                error: action.response,
                teamsUptoDate: true,
            });
        case "ADD_TEAM_INIT":
        case "DELETE_TEAM_INIT":
        case "UPDATE_TEAM_INIT":
            return updateObject(state, {
                closeModal: false,
            });
        case "ADD_TEAM_SUCCESS":
        case "UPDATE_TEAM_SUCCESS":
            return updateObject(state, {
                teamsUptoDate: false,
                closeModal: true,
            });
        case "ADD_TEAM_ERROR":
            return updateObject(state, {
                error: action.response,
            });

        case "DELETE_TEAM_SUCCESS":
            return updateObject(state, {
                teamsUptoDate: false,
                closeModal: true,
            });
        default:
            return state;
    }
}
