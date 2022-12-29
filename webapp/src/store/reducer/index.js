import { combineReducers } from "redux";
import uploadReducer from "store/reducer/uploadReducer";
import scrumTeamNameReducer from "store/reducer/scrumTeamNameReducer";
import addActions from "store/reducer/addActions";
import tabsReducer from "store/reducer/tabsReducer";
import dropDownReducer from "./selectedDropDown";
import loaderReducer from "store/reducer/loaderReducer";
import momReducer from "store/reducer/momReducer";
import insightsReducer from "store/reducer/insights";
import dashboardActionTabReducer from "store/reducer/dashbordActionTab";
import userLoginReducer from "store/reducer/login";
import componentReducer from "store/reducer/components";
import summaryReducer from "store/reducer/summaryReducer";
import configReducer from "store/reducer/configReducer";
import errorReducer from "store/reducer/errorReducer";
import thunk from "redux-thunk";
import { createStore, applyMiddleware } from "redux";
import { composeWithDevTools } from "redux-devtools-extension";

const appReducer = combineReducers({
    // bringin your reducers here
    uploadReducer,
    scrumTeamNameReducer,
    addActions,
    errorReducer,
    tabsReducer,
    dropDownReducer,
    loaderReducer,
    momReducer,
    insightsReducer,
    dashboardActionTabReducer,
    userLoginReducer,
    componentReducer,
    summaryReducer,
    configReducer,
});

const rootReducer = (state, action) => {
    return appReducer(state, action);
};

const middleware = [thunk];

const store = createStore(
    rootReducer,
    composeWithDevTools(applyMiddleware(...middleware))
);

export default store;
