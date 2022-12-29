const initialState = {
    isLoader: false,
};
export default function loaderReducer(state = initialState, action = {}) {
    switch (action.type) {
        case "START_LOADER":
            return { ...state, ...action.payload };
        case "STOP_LOADER":
            return { ...state, ...action.payload };
        default:
            return state;
    }
}
