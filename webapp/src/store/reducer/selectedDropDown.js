const initialState = {
    labels: [],
    sentiments: [],
    assignedTo: [],
    eventDate: [],
};
export default function selectedDropDownReducer(state = initialState, action) {
    switch (action.type) {
        case "ADD_DROPDOWN":
            return { ...state, ...action.payload };
        case "UPDATE_DROPDOWN":
            return { ...state, ...action.payload };
        case "REMOVE_DROPDOWN":
            return action.payload;
        default:
            return state;
    }
}
