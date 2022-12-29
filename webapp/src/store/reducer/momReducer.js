import { updateObject } from "utils/utility";

let meetingData = {
    project_name: "",
    mom_generation_date: "",
    organiser: "",
    location: "",
    meeting_duration: null,
    meeting: { team_name: 1, full_team_name: "" },
    uploaded_date: "",
    meeting_status: "",
};
const initialState = {
    momJsonUptoDate: false,
    momJson: {},
    momStore: {},
    redirection_mask_id: "",
    meetingmetadata: meetingData,
    meetingmetadataUptoDate: true,
    oldJson: {},
    error: false,
    isSaveChanges: false,
};

export default function momReducer(state = initialState, action = {}) {
    switch (action.type) {
        case "UPDATE_MOM_APIS":
            return { ...state, ...action.payload };
        case "SAVE_CHANGES":
            return { ...state, ...action.payload };
        case "GET_MOM_SUCCESS":
            return updateObject(state, {
                momJson: action.response,
                momStore: action.response,
                momJsonUptoDate: true,
            });
        case "GET_MOM_ERROR":
            return updateObject(state, {
                momJson: action.response,
                error: true,
            });
        case "GET_MOM_PROMISEERROR":
            return updateObject(state, {
                momJson: action.response,
                error: true,
            });
        case "UPDATE_FEEDBACK_SUCCESS":
            return updateObject(state, {
                updateFeedbackMessage: action.response,
                momJsonUptoDate: false,
            });
        case "UPDATE_FEEDBACK_ERROR":
            return updateObject(state, {
                updateFeedbackMessage: action.response,
            });
        case "UPDATE_FEEDBACK_PROMISEERROR":
            return updateObject(state, {
                updateFeedbackMessage: action.response,
            });

        case "GET_MEETING_METADATA_SUCCESS":
            return updateObject(state, {
                meetingmetadata: action.response,
                meetingmetadataUptoDate: true,
            });
        case "GET_MEETING_METADATA_PROMISEERROR":
        case "PATCH_MEETING_METADATA_ERROR":
        case "PATCH_MEETING_METADATA_PROMISEERROR":
        case "GET_MEETING_METADATA_ERROR":
            return updateObject(state, {
                meetingmetadata: action.response,
            });
        case "PATCH_MEETING_METADATA_SUCCESS":
            return updateObject(state, {
                meetingmetadata: action.response,
                meetingmetadataUptoDate: false,
            });
        case "CHANGE_FILE_STATUS_SUCCESS":
            return updateObject(state, { meetingmetadataUptoDate: false });
        case "UPDATE_MOM":
            return updateObject(state, {
                momStore: action.payload,
            });
        case "UPDATE_MOM_STORE":
            return updateObject(state, {
                momStore: action.payload.momStore,
            });
        default:
            return state;
    }
}
