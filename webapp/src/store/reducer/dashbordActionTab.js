const initialState = {
    selectedActionRowData: {},
    selectedActionRowIdx: "",
    selectedCardInfo: { cardTitle: "", openCardModal: false },
};
export default function dashboardActionTabReducer(
    state = initialState,
    action
) {
    switch (action.type) {
        case "SELECTED_ACTION_ROW_DATA":
            return { ...state, ...action.payload };
        default:
            return state;
    }
}
