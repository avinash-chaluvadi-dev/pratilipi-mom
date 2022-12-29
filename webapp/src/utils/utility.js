import store from "store/reducer";
import { errorMessage } from "store/action/types";
import { ConstantValue } from "utils/constant";

export const updateObject = (oldObject, updatedValues) => {
    return { ...oldObject, ...updatedValues };
};

export const isEmpty = (obj) => {
    for (var x in obj) {
        if (obj.hasOwnProperty(x)) return false;
    }
    return true;
};

export const handleError = (error) => {
    store.dispatch({
        type: errorMessage["COMMON_ERROR"],
        payload: { showError: true, error: error },
    });
};

export const customErrorMsg = (errorObj) => {
    if (
        errorObj.message?.includes("400") &&
        errorObj.config?.url.includes("/auth/login/")
    )
        return ConstantValue.CUSTOM_ERROR_LOGIN;
    else if (errorObj.message?.includes("400"))
        return ConstantValue.CUSTOM_ERROR_400;
    else if (errorObj.message?.includes("404"))
        return ConstantValue.CUSTOM_ERROR_404;
    else if (errorObj.message?.includes("500"))
        return ConstantValue.CUSTOM_ERROR_500;
    else if (errorObj.message?.includes("502"))
        return ConstantValue.CUSTOM_ERROR_502;
    else if (errorObj.message?.includes("503"))
        return ConstantValue.CUSTOM_ERROR_503;
    else if (errorObj.message?.includes("timeout"))
        return ConstantValue.CUSTOM_ERROR_TIMEOUT;
    else return ConstantValue.CUSTOM_ERROR_COMMON;
};
