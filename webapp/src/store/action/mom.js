import { Mom } from "./types";
import service from "api/service";

export const getMom = (maskedId) => (dispatch) => {
    service.get(dispatch, "/module/mom/" + maskedId + "/", Mom["GET_MOM"]);
};

export const updateFeedback = (maskedId, body) => (dispatch) => {
    service.patchforFLPScreen(
        dispatch,
        "/module/mom/" + maskedId + "/",
        body,
        Mom["UPDATE_FEEDBACK"]
    );
};

export const getMeetingMetaData = (maskedId) => (dispatch) => {
    service.get(
        dispatch,
        "/api/meeting-metadata/" + maskedId + "/",
        Mom["GET_MEETING_METADATA"]
    );
};

export const patchMeetingMetaData = (maskedId, body) => (dispatch) => {
    service.patchforFLPScreen(
        dispatch,
        "/api/meeting-metadata/" + maskedId + "/",
        body,
        Mom["PATCH_MEETING_METADATA"]
    );
};

export const getPDFReport = (maskedId, download) => (dispatch) => {
    service.getWithFileDownload(
        dispatch,
        `/api/mom/report/${maskedId}/`,
        Mom["GET_PDF_REPORT"],
        download
    );
};
