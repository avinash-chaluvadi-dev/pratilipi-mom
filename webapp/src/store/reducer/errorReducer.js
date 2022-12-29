import { updateObject } from "utils/utility";

const initialState = {
    showError: false,
    showErrorMessage: "",
};
export default function errorReducer(state = initialState, action) {
    switch (action.type) {
        case "COMMON_ERROR":
            return updateObject(state, {
                showError: action.payload.showError,
                showErrorMessage: action.payload.error,
            });
        default:
            return state;
    }
}
