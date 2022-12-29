const initialState = [];
export default function editsReducer(state = initialState, action) {
    switch (action.type) {
        case "ADD_ACTION":
            return [state, ...action.payload];
        case "UPDATE_ACTION":
            return { ...state, ...action.payload };
        case "REMOVE_ACTION":
            return action.payload;
        default:
            return state;
    }
}
