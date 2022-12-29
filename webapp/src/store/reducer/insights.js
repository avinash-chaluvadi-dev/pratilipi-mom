const initialState = {
    selectedTab: "",
    selectedIdx: 1,
};
export default function insightReducer(state = initialState, action) {
    switch (action.type) {
        case "SWITCH_INSIGHT_TABS":
            return { ...state, ...action.payload };
        default:
            return state;
    }
}
