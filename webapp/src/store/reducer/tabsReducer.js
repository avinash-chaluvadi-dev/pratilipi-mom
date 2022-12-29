const initialState = {
    title: 0,
    isUserSelection: false,
    userName: "",
    isHilightCard: false,
    highlitedGroup: "",
    from: "",
};
export default function TabsReducer(state = initialState, action = {}) {
    switch (action.type) {
        case "SWITCH_TABS":
            return { ...state, ...action.payload };
        case "UPDATE_TABS":
            return { ...state, ...action.payload };
        case "REMOVE_TABS":
            return action.payload;
        default:
            return state;
    }
}
