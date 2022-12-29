import { Upload } from "./types";
import service from "api/service";

export const uploadFile = (body, setProgress, source) => (dispatch) => {
    service.postForFileUpload(
        dispatch,
        "/api/uploads/",
        body,
        Upload["UPLOAD_FILE"],
        setProgress,
        source
    );
};

export const getAllUploadedFiles = () => (dispatch) => {
    service.get(dispatch, "/api/uploads/", Upload["GET_UPLOADED_FILES"]);
};

export const cancelFileUpload = (id) => (dispatch) => {
    service.patch(
        dispatch,
        "/api/uploads/" + id + "/cancel_extraction/",
        Upload["CANCEL_FILE_UPLOAD"]
    );
};

export const changeFileStatus = (maskedId, body) => (dispatch) => {
    service.patchforFLPScreen(
        dispatch,
        "/api/uploads/" + maskedId + "/",
        body,
        Upload["CHANGE_FILE_STATUS"]
    );
};

export const processUploadedFile = (maskedId) => (dispatch) => {
    service.get(
        dispatch,
        "/api/process-meeting/" + maskedId + "/",
        Upload["PROCESS_UPLOADED_FILE"]
    );
};

export const getUploadFileTrascription = (fileUrl) => (dispatch) => {
    service.getFileTranscribe(
        dispatch,
        `?file_uri=${fileUrl}`,
        Upload["PROCESS_FILE_TRANSCRIPTION"]
    );
};

export const getPlayBackFilePath = (fileUrl) => (dispatch) => {
    service.getPlayBackFile(
        dispatch,
        `?file_uri=${fileUrl}`,
        Upload["PLAYBACK_FILE_URL"]
    );
};

export const getFramesFilePath = (fileUrl) => (dispatch) => {
    service.getPlayBackFile(
        dispatch,
        `?file_uri=${fileUrl}`,
        Upload["FRAMES_FILE_URL"]
    );
};
