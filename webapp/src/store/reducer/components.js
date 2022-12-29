const initialState = {
    selectedItem: 0,
};
export default function componentReducer(state = initialState, action = {}) {
    switch (action.type) {
        case "SWITCH_COMPONENTS":
            return { ...state, ...action.payload };
        default:
            return state;
    }
}
