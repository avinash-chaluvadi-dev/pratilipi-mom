import { ScrumTeamName } from "./types";
import service from "api/service";

export const getScrumTeamName = () => (dispatch) => {
    service.get(dispatch, "/api/teams/", ScrumTeamName["GET_SCRUM_TEAM_NAME"]);
};

export const addScrumTeamName = (body) => (dispatch) => {
    service.post(
        dispatch,
        "/api/teams/",
        body,
        ScrumTeamName["ADD_SCRUM_TEAM_NAME"]
    );
};
