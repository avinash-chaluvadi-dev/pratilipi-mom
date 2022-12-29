import { updateObject } from "utils/utility";

const initialState = {
    uploadedFilesUptoDate: false,
};

export default function uploadReducer(state = initialState, action = {}) {
    switch (action.type) {
        case "UPLOAD_FILE_SUCCESS":
            return updateObject(state, {
                currentUploadedFile: action.response,
                uploadedFilesUptoDate: false,
            });
        case "UPLOAD_FILE_ERROR":
            return updateObject(state, {
                currentUploadedFile: action.response,
            });
        case "UPLOAD_FILE_PROMISEERROR":
            return updateObject(state, {
                currentUploadedFile: action.response,
            });
        case "GET_UPLOADED_FILES_SUCCESS":
            return updateObject(state, {
                uploadedFiles: action.response,
                uploadedFilesUptoDate: true,
            });
        case "GET_UPLOADED_FILES_ERROR":
            return updateObject(state, { uploadedFiles: action.response });
        case "GET_UPLOADED_FILES_PROMISEERROR":
            return updateObject(state, { uploadedFiles: action.response });
        case "CANCEL_FILE_UPLOAD_SUCCESS":
            return updateObject(state, {
                cancelFileUploadResponse: action.response,
                uploadedFilesUptoDate: false,
            });
        case "CANCEL_FILE_UPLOAD_ERROR":
            return updateObject(state, {
                cancelFileUploadResponse: action.response,
            });
        case "CANCEL_FILE_UPLOAD_PROMISEERROR":
            return updateObject(state, {
                cancelFileUploadResponse: action.response,
            });
        case "CHANGE_FILE_STATUS_SUCCESS":
            return updateObject(state, {
                changeFileStatusResponse: action.response,
                uploadedFilesUptoDate: false,
            });
        case "CHANGE_FILE_STATUS_ERROR":
            return updateObject(state, {
                changeFileStatusResponse: action.response,
            });
        case "CHANGE_FILE_STATUS_PROMISEERROR":
            return updateObject(state, {
                changeFileStatusResponse: action.response,
            });
        case "PROCESS_UPLOADED_FILE_SUCCESS":
            return updateObject(state, {
                processUploadedFileResponse: action.response,
                uploadedFilesUptoDate: false,
            });
        case "PROCESS_UPLOADED_FILE_ERROR":
            return updateObject(state, {
                processUploadedFileResponse: action.response,
            });
        case "PROCESS_UPLOADED_FILE_PROMISEERROR":
            return updateObject(state, {
                processUploadedFileResponse: action.response,
            });
        case "PROCESS_FILE_TRANSCRIPTION_SUCCESS":
            return updateObject(state, {
                transcribedFile: action.response,
            });
        case "PROCESS_FILE_TRANSCRIPTION_ERROR":
            return updateObject(state, {
                transcribedFile: action.response,
            });
        case "PLAYBACK_FILE_URL_SUCCESS":
            return updateObject(state, {
                playbackFileUrl: action.response,
            });
        case "PLAYBACK_FILE_URL_ERROR":
            return updateObject(state, {
                playbackFileUrl: action.response,
            });
        case "FRAMES_FILE_URL_SUCCESS":
            return updateObject(state, {
                framesFileUrl: action.response,
            });
        case "FRAMES_FILE_URL_ERROR":
            return updateObject(state, {
                framesFileUrl: action.response,
            });
        default:
            return state;
    }
}
