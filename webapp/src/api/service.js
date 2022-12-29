/* This API wrapper is useful because it:
   1. Centralizes our Axios default configuration.
   2. Abstracts away the logic for determining the baseURL.
   3. Provides a clear, easily consumable list of JavaScript functions
      for interacting with the API. This keeps API calls short and consistent. 
*/

import config from "./config";
import { CommonResponse } from "store/action/types";

const { http, getHeaders, transcribeAPI, playBackAPICall } = config;

const SUCCESSSTATUS = [200, 201, 204];
export const post = async (dispatch, url, data, types) => {
    try {
        dispatch({
            type: types + "_" + CommonResponse["INIT"],
        });
        const response = await http.post(url, data, { headers: getHeaders() });
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const postForFileUpload = async (
    dispatch,
    url,
    data,
    types,
    setProgress,
    source
) => {
    try {
        const response = await http.post(url, data, {
            onUploadProgress: (progress) => {
                const { loaded, total } = progress;
                const percentageProgress = Math.floor((loaded / total) * 100);
                setProgress(percentageProgress);
            },
            cancelToken: source.token,
            headers: getHeaders(),
        });
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const get = async (dispatch, url, types) => {
    try {
        const response = await http.get(url, { headers: getHeaders() });
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const getFileTranscribe = async (dispatch, url, types) => {
    try {
        const response = await transcribeAPI.get(url); //headers not required for direct call of aws lambda function
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const getPlayBackFile = async (dispatch, url, types) => {
    try {
        const response = await playBackAPICall.get(url); //headers not required for direct call of aws lambda function
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const put = async (dispatch, url, data, types) => {
    try {
        const response = await http.put(url, data, { headers: getHeaders() });
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const remove = async (dispatch, url, types) => {
    try {
        dispatch({
            type: types + "_" + CommonResponse["INIT"],
        });
        const response = await http.delete(url, { headers: getHeaders() });
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const patch = async (dispatch, url, data, types) => {
    try {
        dispatch({
            type: types + "_" + CommonResponse["INIT"],
        });
        const response = await http.patch(url, data, { headers: getHeaders() });
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const patchforFLPScreen = async (dispatch, url, data, types) => {
    try {
        const response = await http.patch(url, data, { headers: getHeaders() });
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const getWithFileDownload = async (dispatch, url, types, download) => {
    try {
        const response = await http.get(url, { headers: getHeaders() });
        const object = window.URL.createObjectURL(
            new Blob([response.data], { type: "application/pdf" })
        );
        if (download) {
            const link = document.createElement("a");
            link.href = object;
            link.setAttribute("download", "mom.pdf");
            document.body.appendChild(link);
            link.click();
        } else {
            const pdfWindow = window.open();
            pdfWindow.location.href = object;
        }

        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const getWithParams = async (dispatch, url, params, types) => {
    try {
        let reqBody = {
            method: "get",
            url: url,
            params: params,
            headers: getHeaders(),
        };
        const response = await http(reqBody);
        dispatchResponse(dispatch, response, types);
    } catch (e) {
        errorHandler(e, dispatch, url, types);
    }
};

export const dispatchResponse = async (dispatch, response, types) => {
    try {
        if (SUCCESSSTATUS.includes(response.status)) {
            dispatch({
                type: types + "_" + CommonResponse["SUCCESS"],
                response: await response.data,
            });
            if (types === "AUTHENTICATE_USER_LOGIN") {
                dispatch({
                    type: "AUTHENTICATED",
                    payload: { isUserAuthenticated: true },
                });
            }
        } else {
            dispatch({
                type: types + "_" + CommonResponse["ERROR"],
                response: await response.data,
            });
        }
    } catch (e) {
        dispatch({
            type: types + "_" + CommonResponse["PROMISEERROR"],
            response: await response.data,
        });
        return e;
    }
};

const errorHandler = (e_obj, dispatch, url, types) => {
    if (!e_obj.status) {
        dispatch({
            type: types + "_" + CommonResponse["PROMISEERROR"],
            response: { status: e_obj },
        });
        return;
    }
    dispatch({
        type: CommonResponse["PROMISEERROR"] + "_" + types,
        response: e_obj,
    });
    return e_obj;
};

const APIService = {
    get,
    post,
    postForFileUpload,
    put,
    remove,
    patch,
    patchforFLPScreen,
    getWithFileDownload,
    getWithParams,
    getFileTranscribe,
    getPlayBackFile,
};
export default APIService;
