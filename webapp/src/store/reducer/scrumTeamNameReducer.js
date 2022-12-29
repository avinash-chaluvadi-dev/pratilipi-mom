import { updateObject } from "utils/utility";

let TeamLoggoff = [];
const initialState = {
    scrumTeamNameUptoDate: false,
    scrumTeams: TeamLoggoff,
};

export default function scrumTeamNameReducer(
    state = initialState,
    action = {}
) {
    switch (action.type) {
        case "GET_SCRUM_TEAM_NAME_SUCCESS":
            return updateObject(state, {
                scrumTeams: action.response,
                scrumTeamNameUptoDate: true,
            });
        case "GET_SCRUM_TEAM_NAME_ERROR":
            return updateObject(state, { scrumTeams: action.response });
        case "GET_SCRUM_TEAM_NAME_PROMISEERROR":
            return updateObject(state, { scrumTeams: action.response });
        case "ADD_SCRUM_TEAM_NAME_SUCCESS":
            return updateObject(state, {
                currentScrumTeam: action.response,
                scrumTeamNameUptoDate: false,
            });
        case "ADD_SCRUM_TEAM_NAME_ERROR":
            return updateObject(state, { currentScrumTeam: action.response });
        case "ADD_SCRUM_TEAM_NAME_PROMISEERROR":
            return updateObject(state, { currentScrumTeam: action.response });
        default:
            return state;
    }
}
