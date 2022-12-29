import axios from "axios";
import { ConstantValue } from "utils/constant";
import { handleError } from "../utils/utility";

let domainUrl = window.location.href;
let url = new URL(domainUrl);
let transcribeUrl = ConstantValue.TRANSCRIBE_URL;
let playbackUrl = ConstantValue.PLAYBACK_URL;
let timeoutval = ConstantValue.TIMEOUT_VALUE;

const http = axios.create({
    baseURL: `${url.origin}${ConstantValue.API_SERVICE_PATH_NAME}`,
    timeout: `${timeoutval}`,
});

const transcribeAPI = axios.create({
    baseURL: `${transcribeUrl}`,
    timeout: `${timeoutval}`,
});

const playBackAPICall = axios.create({
    baseURL: `${playbackUrl}`,
    timeout: `${timeoutval}`,
});

function getAuthToken() {
    return "Token " + localStorage.getItem("accessToken");
}

const getHeaders = () => {
    const obj = { "Content-type": "application/json" };

    if (localStorage.getItem("accessToken"))
        obj["Authorization"] = getAuthToken();

    return obj;
};

http.interceptors.response.use(
    (response) => response,
    (error) => {
        handleError(error);
    }
);

transcribeAPI.interceptors.response.use(
    (response) => response,
    (error) => {
        handleError(error);
    }
);

playBackAPICall.interceptors.response.use(
    (response) => response,
    (error) => {
        handleError(error);
    }
);

const config = { http, getHeaders, transcribeAPI, playBackAPICall };
export default config;
