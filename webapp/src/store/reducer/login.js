import { updateObject } from "utils/utility";

const initialState = {
    userAuthenticated: false,
    accessToken: "",
    userName: "",
    role: {},
    isUserAuthenticated: false,
};

let auth = localStorage.getItem("userAuthenticated");
initialState.userAuthenticated = auth ? auth : false;
initialState.accessToken = localStorage.getItem("accessToken");
initialState.userName = localStorage.getItem("userName");
initialState.role = JSON.parse(localStorage.getItem("userRole"));

export default function userLoginReducer(state = initialState, action = {}) {
    switch (action.type) {
        case "AUTHENTICATE_USER_LOGIN_SUCCESS":
            localStorage.setItem("userAuthenticated", true);
            localStorage.setItem("accessToken", action.response.token);
            localStorage.setItem("userName", action.response.name);
            localStorage.setItem("email", action.response.email);
            localStorage.setItem(
                "userRole",
                JSON.stringify(action.response.role)
            );
            return updateObject(state, {
                authenticationMessage: action.response.message,
                accessToken: action.response.token,
                userAuthenticated: true,
                role: action.response.role,
            });
        case "AUTHENTICATE_USER_LOGIN_ERROR":
            return updateObject(state, {
                authenticationMessage: action.response,
            });
        case "AUTHENTICATE_USER_LOGIN_PROMISEERROR":
            return updateObject(state, {
                authenticationMessage: action.response,
            });
        case "AUTHENTICATED":
            return updateObject(state, { isUserAuthenticated: true });
        default:
            return state;
    }
}
