import { Config } from "./types";
import service from "api/service";

export const getAllTeams = () => (dispatch) => {
    service.get(dispatch, "/api/teams/", Config["GET_ALL_TEAMS"]);
};

export const addTeam = (body) => (dispatch) => {
    service.post(dispatch, "/api/teams/", body, Config["ADD_TEAM"]);
};

export const deleteTeam = (id) => (dispatch) => {
    service.remove(dispatch, `/api/teams/${id}/`, Config["DELETE_TEAM"]);
};

export const updateTeam = (id, body) => (dispatch) => {
    service.patch(dispatch, `/api/teams/${id}/`, body, Config["UPDATE_TEAM"]);
};
