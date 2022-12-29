import { findAll } from "highlight-words-core";

const searchNames = [
    "bhanu",
    "himanshu",
    "kajari",
    "divesh",
    "sravanthi",
    "avinash",
    "hrisheek",
    "avijit",
    "amit",
    "silpa",
    "sarika",
    "[@]twenty2[#]22[#][@]",
];
const searchTechs = [
    "Deployment",
    "aws",
    "Cortex",
    "WorkOS",
    "machine learning",
    "Supervised",
    "Unsupervised",
];
const searchTools = [
    "jira",
    "team's",
    "Excel Sheet",
    "ppt",
    "Outlook",
    "microsoft word",
    "MS-word",
    "power point presentation",
    "webex",
];
const searchAnthemTools = [
    "BCP",
    "Service Now",
    "Pulse",
    "IT Desk",
    "IT Service connect",
    "IT enterprise service desk",
    "ADP etime",
    "LMS",
    "Learning Management System",
    "Taleo",
    "Impact Recognition",
];

function HighlightWord(props) {
    const textToHighlight = props.texttohighlight;
    const errorToHighlight = props.errorToHighlight;
    var error_0 = [];
    var error_1 = [];
    var error_2 = [];
    var error_3 = [];
    var error_4 = [];
    var error_5 = [];
    var error_6 = [];
    var error_7 = [];
    var error_8 = [];
    var error_9 = [];
    var error_10 = [];
    var error_11 = [];
    var error_12 = [];
    var error_13 = [];
    var error_14 = [];
    var error_15 = [];

    for (let key in errorToHighlight) {
        if (errorToHighlight[key]["errorCode"] == 0) {
            error_0 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 1) {
            error_1 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 2) {
            error_2 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 3) {
            error_3 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 4) {
            error_4 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 5) {
            error_5 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 6) {
            error_6 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 7) {
            error_7 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 8) {
            error_8 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 9) {
            error_9 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 10) {
            error_10 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 11) {
            error_11 = errorToHighlight[key]["errorText"];
            //console.log(error_11)
        }
        if (errorToHighlight[key]["errorCode"] == 12) {
            error_12 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 13) {
            error_13 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 14) {
            error_14 = errorToHighlight[key]["errorText"];
        }
        if (errorToHighlight[key]["errorCode"] == 15) {
            error_15 = errorToHighlight[key]["errorText"];
        }
    }

    const searchWords = searchNames.concat(
        searchTechs,
        searchTools,
        searchAnthemTools,
        error_0,
        error_1,
        error_2,
        error_3,
        error_4,
        error_5,
        error_6,
        error_7,
        error_8,
        error_9,
        error_10,
        error_11,
        error_12,
        error_13,
        error_14,
        error_15
    );

    const chunksAll = findAll({
        autoEscape: true,
        searchWords: searchWords,
        textToHighlight: textToHighlight,
    });

    let high = chunksAll.map((chunk) => {
        const { end, highlight, start } = chunk;
        const text = textToHighlight.substr(start, end - start);
        if (highlight) {
            if (error_0.includes(text.toLowerCase())) {
                return <mark className="HighlightError0">{text}</mark>;
            } else if (error_1.includes(text.toLowerCase())) {
                return <mark className="HighlightError1">{text}</mark>;
            } else if (error_2.includes(text.toLowerCase())) {
                return <mark className="HighlightError2">{text}</mark>;
            } else if (error_3.includes(text.toLowerCase())) {
                return <mark className="HighlightError3">{text}</mark>;
            } else if (error_4.includes(text.toLowerCase())) {
                return <mark className="HighlightError4">{text}</mark>;
            } else if (error_5.includes(text.toLowerCase())) {
                return <mark className="HighlightError5">{text}</mark>;
            } else if (error_6.includes(text.toLowerCase())) {
                return <mark className="HighlightError6">{text}</mark>;
            } else if (error_7.includes(text.toLowerCase())) {
                return <mark className="HighlightError7">{text}</mark>;
            } else if (error_8.includes(text.toLowerCase())) {
                return <mark className="HighlightError8">{text}</mark>;
            } else if (error_9.includes(text.toLowerCase())) {
                return <mark className="HighlightError9">{text}</mark>;
            } else if (error_10.includes(text.toLowerCase())) {
                return <mark className="HighlightError10">{text}</mark>;
            } else if (error_11.includes(text.toLowerCase())) {
                return <mark className="HighlightError11">{text}</mark>;
            } else if (error_12.includes(text.toLowerCase())) {
                return <mark className="HighlightError12">{text}</mark>;
            } else if (error_13.includes(text.toLowerCase())) {
                return <mark className="HighlightError13">{text}</mark>;
            } else if (error_14.includes(text.toLowerCase())) {
                return <mark className="HighlightError14">{text}</mark>;
            } else if (error_15.includes(text.toLowerCase())) {
                return <mark className="HighlightError15">{text}</mark>;
            } else if (searchNames.includes(text.toLowerCase())) {
                return <mark className="HighlightClassNames">{text}</mark>;
            } else if (searchTechs.includes(text.toLowerCase())) {
                return <mark className="HighlightClassTechs">{text}</mark>;
            } else if (searchTools.includes(text.toLowerCase())) {
                return <mark className="HighlightClassTools">{text}</mark>;
            } else {
                return (
                    <mark className="HighlightClassAnthemTools">{text}</mark>
                );
            }
        } else {
            return text;
        }
    });
    return <div className="summary-highlighted">{high}</div>;
}

export default HighlightWord;
