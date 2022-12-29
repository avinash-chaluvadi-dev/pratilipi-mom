import { Login } from "./types";
import service from "api/service";

export const userLogin = (body) => (dispatch) => {
    service.post(
        dispatch,
        "/auth/login/",
        body,
        Login["AUTHENTICATE_USER_LOGIN"]
    );
};
