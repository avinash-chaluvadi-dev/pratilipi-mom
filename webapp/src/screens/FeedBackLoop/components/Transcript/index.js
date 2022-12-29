import React from "react";
import ReactDOM from "react-dom";
import HighlightWord from "./HighlightWords";
import axios from "axios";
import { TokenAnnotator, TextAnnotator } from "react-text-annotate";

var jsonData = {};

var errorCodes = {
    "[]": 0,
    "[-]": 1,
    "[.]": 2,
    "[**]": 3,
    "[$]": 4,
    "[#]": 5,
    "()": 6,
    "[??]": 7,
    "[=]": 8,
    "[&]": 9,
    "[%]": 10,
    "[@]": 11,
    "<>": 12,
    "(*)": 13,
    "[/]": 14,
    "[sc]": 15,
};
var errorMeanings = {
    "[]": "Extra words by machine",
    "[-]": "missed word by machine.",
    "[.]": "Speaker Separation. Two speaker conversation. ",
    "[**]": "Word or sentence not clear during the conversation / speech. ",
    "[$]": "Machine totally skipped to transcribe the sentence",
    "[#]": "Words/ sentences being interpreted incorrectly ",
    "()": "sentences being interpreted incorrectly ",
    "[??]": "speech is unintelligible, making an educated guess",
    "[=]": "when two speaker speak directly after one another without a gap/ Fast reply",
    "[&]": "When two speaker conversation overlaps. ",
    "[%]": "laughter",
    "[@]": "number pronounced as words {4 as four}",
    "<>": "Words interpreted incorrectly ",
    "(*)": "speech is unintelligible, making an educated guess",
    "[PI]": "Personal Information /Personal Data. ",
    "[sc]": "Word level spell correction",
    "[sp]": "Speaker Separation",
};

var errorFormat = {
    "[]": "[] word/sentence [/]",
    "[-]": "[-] word/sentence [/-]",
    "[.]": "[.] word/sentence [/.]",
    "[**]": "Sentence [**]",
    "[$]": "[$] word/sentence [/$]",
    "[#]": "Part of Error 6 and Error 12",
    "()": "() wrong sentence [#] Correct sentence[/#] (/)",
    "[??]": "Part of Error 13",
    "[=]": "[=] word/sentence [/=]",
    "[&]": "[&] word/sentence [/&]",
    "[%]": "[%]",
    "[@]": "[@] wrong output [#] Correct output[/#][/@]",
    "<>": "<> wrong word [#] correct word [/#]</>",
    "(*)": "(*) misinterpreted sentence [??] educated/intelligent guess [/??] (/*)",
    "[PI]": "[PI] Word/ Sentence [/PI]",
    "[sc]": "[sc] wrong word [#] correct word[/#][/sc]",
};

let testString =
    "Okay.[] So[]. Today I have to []to[] < did [#] do [#] > all of () this task [#] that work [#] () .(*) Dhanuj [??] Anuj [??] (*). Will you do it ?. [-] Yes .[-]";

var changesMadeNER = new Set();
var changesMade = new Array();

var NERchangesJsonData = [];
var errorRulesJsonData = [];

const divStyle = {
    color: "rgb(0, 0, 0)",
    "background-color": "rgba(255, 255, 255, 0.1)",
    width: "100%",
    margin: "auto",
    padding: "10px",
};

const TEXT =
    "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously. “I can tell you very senior CEOs of major American car companies would shake my hand and turn away because I wasn’t worth talking to,” said Thrun, now the co-founder and CEO of online higher education startup Udacity, in an interview with Recode earlier this week. A little less than a decade later, dozens of self-driving startups have cropped up while automakers around the world clamor, wallet in hand, to secure their place in the fast-moving world of fully automated transportation.";
const TAG_COLORS = {
    NAME: "#00ffa2",
    TOOL: "#84d2ff",
    ANTHEMTOOL: "#d7f06b",
    TECHNOLOGY: " #eeac55",
};

const Card = ({ children }) => (
    <div
        style={{
            boxShadow: "0 2px 4px rgba(0,0,0,.1)",
            margin: 6,
            maxWidth: 500,
            padding: 16,
        }}
    >
        {children}
    </div>
);

class TranscriptDashboard extends React.Component {
    state = {
        meeting: [],
        meetingInfo: [],
        new_transcription: [],
        editing: false,
        NumberHolder: 1,
    };

    componentDidMount() {
        fetch("./data/main.json")
            .then((resp) => resp.json())
            .then((data) => {
                jsonData = data;
                this.setState({
                    meeting: data[0].transcriptions,
                    meetingInfo: data[0].meetingInfo[0],
                    title: data[0].meetingInfo[0].title,
                });
            });
    }

    onTranscriptChange = (change) => {
        this.setState({
            new_transcription: change,
        });
    };

    removeExtraSpaces = (sentence) => {
        return sentence.replace(/\s+/g, " ");
    };
    removeExtraRules = (sentence) => {
        return sentence.replaceAll("[%]", "").replaceAll("[**]", "");
        // replaceAll("[.]","").replaceAll("[&]","").replaceAll("[=]","")
    };

    intermediateFunc = (sentence, i, rule) => {
        let correctString = "";
        let step = rule.length;
        while (sentence.slice(i, i + step) !== rule && i < sentence.length) {
            correctString += sentence.charAt(i);
            i += 1;
        }
        if (sentence.slice(i, i + step) === rule) {
            i += step;
        }
        return [correctString, i];
    };

    processRules = (sentence) => {
        sentence = this.removeExtraSpaces(sentence);
        var correctedString = "";
        var i = 0;
        var errorDict = {};
        while (i < sentence.length) {
            if (sentence.slice(i, i + 2) === "[]") {
                i += 2;
                let ruleBase = "[]";
                let endRuleBase = "[/]";
                let wrongText = "";
                while (
                    sentence.slice(i, i + 3) !== endRuleBase &&
                    i < sentence.length
                ) {
                    wrongText += sentence.charAt(i);
                    i += 1;
                }
                if (sentence.slice(i, i + 3) === endRuleBase) {
                    i += 3;
                }
                if (ruleBase in errorDict) {
                    let arr = errorDict[ruleBase]["errorText"];
                    arr.push(ruleBase + wrongText + endRuleBase);
                    errorDict[ruleBase] = {
                        errorCode: errorCodes[ruleBase],
                        errorMeaning: errorMeanings[ruleBase],
                        errorText: arr,
                        errorFormat: errorFormat[ruleBase],
                    };
                } else {
                    errorDict[ruleBase] = {
                        errorCode: errorCodes[ruleBase],
                        errorMeaning: errorMeanings[ruleBase],
                        errorText: [ruleBase + wrongText + endRuleBase],
                        errorFormat: errorFormat[ruleBase],
                    };
                }
                continue;
            }
            if (sentence.slice(i, i + 3) === "[-]") {
                let ruleBase = "[-]";
                let endRuleBase = "[/-]";
                let correctString = "";
                i += 3;
                [correctString, i] = this.intermediateFunc(
                    sentence,
                    i,
                    endRuleBase
                );
                if (ruleBase in errorDict) {
                    let arr = errorDict[ruleBase]["errorText"];
                    arr.push(ruleBase + correctString + endRuleBase);
                    errorDict[ruleBase] = {
                        errorCode: errorCodes[ruleBase],
                        errorMeaning: errorMeanings[ruleBase],
                        errorText: arr,
                        errorFormat: errorFormat[ruleBase],
                    };
                } else {
                    errorDict[ruleBase] = {
                        errorCode: errorCodes[ruleBase],
                        errorMeaning: errorMeanings[ruleBase],
                        errorText: [ruleBase + correctString + endRuleBase],
                        errorFormat: errorFormat[ruleBase],
                    };
                }

                correctedString += correctString;
                continue;
            }

            if (sentence.slice(i, i + 3) === "[$]") {
                let ruleBase = "[$]";
                let endRuleBase = "[/$]";
                let correctString = "";
                i += 3;
                [correctString, i] = this.intermediateFunc(
                    sentence,
                    i,
                    endRuleBase
                );
                if (ruleBase in errorDict) {
                    let arr = errorDict[ruleBase]["errorText"];
                    arr.push(ruleBase + correctString + endRuleBase);
                    errorDict[ruleBase] = {
                        errorCode: errorCodes[ruleBase],
                        errorMeaning: errorMeanings[ruleBase],
                        errorText: arr,
                        errorFormat: errorFormat[ruleBase],
                    };
                } else {
                    errorDict[ruleBase] = {
                        errorCode: errorCodes[ruleBase],
                        errorMeaning: errorMeanings[ruleBase],
                        errorText: [ruleBase + correctString + endRuleBase],
                        errorFormat: errorFormat[ruleBase],
                    };
                }
                correctedString += correctString;
                continue;
            }

            if (sentence.slice(i, i + 3) === "[%]") {
                errorDict["isLaughter"] = true;
                i += 3;
            }
            if (sentence.slice(i, i + 4) === "[**]") {
                let firstIndex = sentence.slice(i, sentence.length);
                if (firstIndex == -1) {
                    firstIndex = 0;
                }
                let lastIndex = sentence.slice(0, i + 1).lastIndexOf(".");
                if (lastIndex == -1) {
                    lastIndex = sentence.length - 1;
                }
                // let sentenceWithError = sentence.slice(firstIndex,lastIndex+1)
                // let arr = [];
                // if("isUnclearWordsOrSentences" in errorDict){
                //   arr = errorDict["isUnclearWordsOrSentences"];
                // }
                // arr.push()
                // errorDict["isUnclearWordsOrSentences"]
                i += 4;
            }

            if (sentence.slice(i, i + 2) === "<>") {
                var prevI = i;
                let ruleBase = "[#]";
                let endRuleBase = "[/#]";
                let secondRuleBase = "<>";
                let endSecondRuleBase = "</>";
                i += 2;
                let wrongString = "";
                while (
                    sentence.slice(i, i + 3) !== ruleBase &&
                    i < sentence.length
                ) {
                    wrongString += sentence.charAt(i);
                    i += 1;
                }
                if (sentence.slice(i, i + 3) === ruleBase) {
                    i += 3;
                    let correctString = "";
                    [correctString, i] = this.intermediateFunc(
                        sentence,
                        i,
                        endRuleBase
                    );
                    let extraString = "";
                    extraString += endRuleBase;
                    while (
                        sentence.slice(i, i + 3) !== endSecondRuleBase &&
                        i < sentence.length
                    ) {
                        extraString += sentence.charAt(i);
                        i += 1;
                    }

                    if (sentence.slice(i, i + 3) === endSecondRuleBase) {
                        extraString += sentence.slice(i, i + 3);
                        i += 3;
                    }
                    if (secondRuleBase + ruleBase in errorDict) {
                        let arr =
                            errorDict[secondRuleBase + ruleBase]["errorText"];
                        arr.push(
                            secondRuleBase +
                                wrongString +
                                ruleBase +
                                correctString +
                                extraString
                        );
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: arr,
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    } else {
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: [
                                secondRuleBase +
                                    wrongString +
                                    ruleBase +
                                    correctString +
                                    extraString,
                            ],
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    }
                    correctedString += correctString;
                } else {
                    correctedString += sentence.charAt(prevI);
                    i = prevI + 1;
                }
                continue;
            }

            if (sentence.slice(i, i + 2) === "()") {
                i += 2;
                let wrongString = "";
                let ruleBase = "[#]";
                let endRuleBase = "[/#]";
                let secondRuleBase = "()";
                let endSecondRuleBase = "(/)";
                while (
                    sentence.slice(i, i + 3) !== ruleBase &&
                    i < sentence.length
                ) {
                    wrongString += sentence.charAt(i);
                    i += 1;
                }
                if (sentence.slice(i, i + 3) === ruleBase) {
                    i += 3;
                    let correctString = "";
                    [correctString, i] = this.intermediateFunc(
                        sentence,
                        i,
                        endRuleBase
                    );
                    let extraString = "";
                    extraString += endRuleBase;

                    while (
                        sentence.slice(i, i + 3) !== endSecondRuleBase &&
                        i < sentence.length
                    ) {
                        extraString += sentence.charAt(i);
                        i += 1;
                    }
                    if (sentence.slice(i, i + 3) === endSecondRuleBase) {
                        extraString += sentence.slice(i, i + 3);
                        i += 3;
                    }
                    if (secondRuleBase + ruleBase in errorDict) {
                        let arr =
                            errorDict[secondRuleBase + ruleBase]["errorText"];
                        arr.push(
                            secondRuleBase +
                                wrongString +
                                ruleBase +
                                correctString +
                                extraString
                        );
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: arr,
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    } else {
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: [
                                secondRuleBase +
                                    wrongString +
                                    ruleBase +
                                    correctString +
                                    extraString,
                            ],
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    }
                    correctedString += correctString;
                }
                continue;
            }

            if (sentence.slice(i, i + 3) === "(*)") {
                i += 3;
                let wrongString = "";
                let ruleBase = "[??]";
                let endRuleBase = "[/??]";
                let secondRuleBase = "(*)";
                let endSecondRuleBase = "(/*)";
                while (
                    sentence.slice(i, i + 4) !== ruleBase &&
                    i < sentence.length
                ) {
                    wrongString += sentence.charAt(i);
                    i += 1;
                }
                if (sentence.slice(i, i + 4) === ruleBase) {
                    i += 4;
                    let correctString = "";
                    [correctString, i] = this.intermediateFunc(
                        sentence,
                        i,
                        endRuleBase
                    );
                    let extraString = "";
                    extraString += endRuleBase;
                    while (
                        sentence.slice(i, i + 4) !== endSecondRuleBase &&
                        i < sentence.length
                    ) {
                        extraString += sentence.charAt(i);
                        i += 1;
                    }
                    if (sentence.slice(i, i + 4) === endSecondRuleBase) {
                        extraString += sentence.slice(i, i + 4);
                        i += 4;
                    }
                    if (secondRuleBase + ruleBase in errorDict) {
                        let arr =
                            errorDict[secondRuleBase + ruleBase]["errorText"];
                        arr.push(
                            secondRuleBase +
                                wrongString +
                                ruleBase +
                                correctString +
                                extraString
                        );
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: arr,
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    } else {
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: [
                                secondRuleBase +
                                    wrongString +
                                    ruleBase +
                                    correctString +
                                    extraString,
                            ],
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    }
                    correctedString += correctString;
                }
                continue;
            }

            if (sentence.slice(i, i + 3) === "[@]") {
                i += 3;
                let wrongString = "";
                let ruleBase = "[#]";
                let endRuleBase = "[/#]";
                let secondRuleBase = "[@]";
                let endSecondRuleBase = "[/@]";
                while (
                    sentence.slice(i, i + 3) !== ruleBase &&
                    i < sentence.length
                ) {
                    wrongString += sentence.charAt(i);
                    i += 1;
                }
                if (sentence.slice(i, i + 3) === ruleBase) {
                    i += 3;
                    let correctString = "";
                    [correctString, i] = this.intermediateFunc(
                        sentence,
                        i,
                        endRuleBase
                    );
                    let extraString = "";
                    extraString += endRuleBase;
                    while (
                        sentence.slice(i, i + 4) !== endSecondRuleBase &&
                        i < sentence.length
                    ) {
                        extraString += sentence.charAt(i);
                        i += 1;
                    }
                    if (sentence.slice(i, i + 4) === endSecondRuleBase) {
                        extraString += sentence.slice(i, i + 4);
                        i += 4;
                    }
                    if (secondRuleBase + ruleBase in errorDict) {
                        let arr =
                            errorDict[secondRuleBase + ruleBase]["errorText"];
                        arr.push(
                            secondRuleBase +
                                wrongString +
                                ruleBase +
                                correctString +
                                extraString
                        );
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: arr,
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    } else {
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: [
                                secondRuleBase +
                                    wrongString +
                                    ruleBase +
                                    correctString +
                                    extraString,
                            ],
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    }
                    correctedString += correctString;
                }
                continue;
            }

            if (sentence.slice(i, i + 4) === "[sc]") {
                i += 4;
                let wrongString = "";
                let ruleBase = "[#]";
                let endRuleBase = "[/#]";
                let secondRuleBase = "[sc]";
                let endSecondRuleBase = "[/sc]";
                while (
                    sentence.slice(i, i + 3) !== ruleBase &&
                    i < sentence.length
                ) {
                    wrongString += sentence.charAt(i);
                    i += 1;
                }
                if (sentence.slice(i, i + 3) === ruleBase) {
                    i += 3;
                    let correctString = "";
                    [correctString, i] = this.intermediateFunc(
                        sentence,
                        i,
                        endRuleBase
                    );
                    let extraString = "";
                    extraString += endRuleBase;
                    while (
                        sentence.slice(i, i + 5) !== endSecondRuleBase &&
                        i < sentence.length
                    ) {
                        extraString += sentence.charAt(i);
                        i += 1;
                    }
                    if (sentence.slice(i, i + 5) === endSecondRuleBase) {
                        extraString += sentence.slice(i, i + 5);
                        i += 5;
                    }
                    if (secondRuleBase + ruleBase in errorDict) {
                        let arr =
                            errorDict[secondRuleBase + ruleBase]["errorText"];
                        let spellCorrectionArr =
                            errorDict["wordLevelSpellCorrection"];
                        arr.push(
                            secondRuleBase +
                                wrongString +
                                ruleBase +
                                correctString +
                                extraString
                        );
                        spellCorrectionArr.push({
                            wrongWord: wrongString,
                            correctWord: correctString,
                        });
                        errorDict["wordLevelSpellCorrection"] =
                            spellCorrectionArr;
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: arr,
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    } else {
                        errorDict["wordLevelSpellCorrection"] = [
                            {
                                wrongWord: wrongString,
                                correctWord: correctString,
                            },
                        ];
                        errorDict[secondRuleBase + ruleBase] = {
                            errorCode: errorCodes[secondRuleBase],
                            errorMeaning: errorMeanings[secondRuleBase],
                            errorText: [
                                secondRuleBase +
                                    wrongString +
                                    ruleBase +
                                    correctString +
                                    extraString,
                            ],
                            errorFormat: errorFormat[secondRuleBase],
                        };
                    }
                    correctedString += correctString;
                }
                continue;
            }

            correctedString += sentence.charAt(i);
            i += 1;
        }
        correctedString = correctedString.replace(/\s+/g, " ");
        correctedString = this.removeExtraRules(correctedString);
        // if ('wordLevelSpellCorrection' in errorDict) {
        //   console.log(
        //     'Word Level Spell Correction:',
        //     errorDict['wordLevelSpellCorrection']
        //   );
        // }
        //console.log(errorDict);
        return [errorDict, correctedString];
    };
    speakerSeparationRule = (text) => {
        let i = 0;
        let finalCorrectedText = "";
        let position = 0;
        let mainRuleBase = "[sp";
        let secondRuleBase = "[=]";
        let secondRuleBasePartTwo = "[/=]";
        let thirdRuleBase = "[&]";
        let thirdRuleBasePartTwo = "[/&]";
        let speakerDict = {};
        text = text
            .replaceAll(secondRuleBase, "")
            .replaceAll(thirdRuleBase, "")
            .replaceAll(secondRuleBasePartTwo, "")
            .replaceAll(thirdRuleBasePartTwo, "");
        while (i < text.length) {
            if (
                text.slice(i, i + 3) == secondRuleBase ||
                text.slice(i, i + 3) == thirdRuleBase
            ) {
                i += 3;
                continue;
            }
            if (text.slice(i, i + 3) == mainRuleBase) {
                position += 1;
                i += 3;
                let speakerNum = "";
                while (i < text.length && text.charAt(i) != "]") {
                    speakerNum += text.charAt(i);
                    i += 1;
                }
                i += 1;
                let speakerText = "";
                while (text.slice(i, i + 3) != mainRuleBase) {
                    speakerText += text.charAt(i);
                    finalCorrectedText += text.charAt(i);
                    i += 1;
                }
                if (text.slice(i, i + 3) == mainRuleBase) {
                    i += 3;
                }
                while (i < text.length && text.charAt(i) != "]") {
                    i += 1;
                }
                let speakerId = "Speaker " + speakerNum;
                if (speakerId in speakerDict) {
                    speakerDict[speakerId].push({
                        speakerText: speakerText,
                        speakerPosition: position,
                    });
                } else {
                    speakerDict[speakerId] = [
                        { speakerText: speakerText, speakerPosition: position },
                    ];
                }
            }
            finalCorrectedText += " ";
            i += 1;
        }
        finalCorrectedText = this.removeExtraSpaces(finalCorrectedText);
        speakerDict["originalText"] = text;
        speakerDict["correctedText"] = finalCorrectedText;
        //console.log('SPEAKER SEPARATION DICTIONARY', speakerDict);
        return [speakerDict, finalCorrectedText];
    };

    errorFind = () => {
        let errorDict = {};
        if (changesMade.length > 0) {
            for (var i = 0; i < changesMade.length; i++) {
                [
                    errorDict,
                    jsonData[0].transcriptions[changesMade[i]]["correctedText"],
                ] = this.processRules(
                    jsonData[0].transcriptions[changesMade[i]].text
                );
                if (
                    jsonData[0].transcriptions[changesMade[i]][
                        "correctedText"
                    ].includes("[sp")
                ) {
                    let speakerDict = {};
                    [
                        speakerDict,
                        jsonData[0].transcriptions[changesMade[i]][
                            "correctedText"
                        ],
                    ] = this.speakerSeparationRule(
                        jsonData[0].transcriptions[changesMade[i]][
                            "correctedText"
                        ]
                    );
                    errorDict["multipleSpeakerDict"] = speakerDict;
                }
                if (Object.keys(errorDict).length !== 0) {
                    jsonData[0].transcriptions[changesMade[i]]["errorDetails"] =
                        errorDict;
                    errorDict["id"] = changesMade[i];
                    errorDict["correctedText"] =
                        jsonData[0].transcriptions[changesMade[i]][
                            "correctedText"
                        ];

                    if (
                        errorRulesJsonData.some((e) => e.id == changesMade[i])
                    ) {
                        errorRulesJsonData = errorRulesJsonData.filter(
                            (item) => item.id !== changesMade[i]
                        );
                        errorRulesJsonData.push(errorDict);
                    } else {
                        errorRulesJsonData.push(errorDict);
                    }
                }

                jsonData[0].transcriptions[changesMade[i]]["text"] =
                    this.removeExtraSpaces(
                        jsonData[0].transcriptions[changesMade[i]]["text"]
                    );
                jsonData[0].transcriptions[changesMade[i]]["text"] =
                    this.removeExtraRules(
                        jsonData[0].transcriptions[changesMade[i]]["text"]
                    );
            }
        }
    };

    handleSubmit = (e) => {
        jsonData = JSON.parse(sessionStorage.getItem("updatedData"))
            ? JSON.parse(sessionStorage.getItem("updatedData"))
            : jsonData;
        this.setState({ editing: false });
        this.errorFind();
        this.NERchanges();
        // axios.post('http://10.124.127.166:5555/insert_vocab', NERchangesJsonData);
        // for (let i = 0; i < changesMade.length; i++) {
        //   console.log(
        //     'CorrectedText for Transcript ' + changesMade[i] + ':',
        //     jsonData[0].transcriptions[changesMade[i]]['correctedText']
        //   );
        // }
        changesMade = [];
        jsonData[0]["errorRules"] = errorRulesJsonData;
        jsonData[0]["nerChanges"] = NERchangesJsonData;
        sessionStorage.setItem("updatedData", JSON.stringify(jsonData));
        //console.log(jsonData);
        // axios
        //   .post('http://10.124.127.166:5555/save', JSON.stringify(jsonData), { headers: { 'Content-Type': 'application/json' } })
        //   .then(() => console.log('JSON Saved'))
        //   .catch(err => {
        //     console.error(err);
        //   });
    };

    NERchanges = () => {
        for (let key of changesMadeNER) {
            var edited;
            if ("correctedText" in jsonData[0].transcriptions[key]) {
                edited = jsonData[0].transcriptions[key].correctedText;
            } else {
                edited = jsonData[0].transcriptions[key].text;
            }

            if (/\{(.*?)\}/.test(edited)) {
                const matches = edited.match(/\{(.*?)\}/g);
                if (NERchangesJsonData.some((e) => e.id == key)) {
                    NERchangesJsonData = NERchangesJsonData.filter(
                        (item) => item.id !== key
                    );
                    this.pushtoNERJson(key, matches[0].toString());
                } else {
                    this.pushtoNERJson(key, matches[0].toString());
                }
                //jsonData[0].transcriptions[key].text = edited.replace(/\{(.*?)\}/g, "")

                if (matches[0].toString().includes("deadline:Y")) {
                    jsonData[0].transcriptions[key].actions.deadline.status =
                        "Y";
                }
                if (matches[0].toString().includes("deadline:N")) {
                    jsonData[0].transcriptions[key].actions.deadline.status =
                        "N";
                }
                if (matches[0].toString().includes("escalation:Y")) {
                    jsonData[0].transcriptions[key].actions.escalation.status =
                        "Y";
                }
                if (matches[0].toString().includes("escalation:N")) {
                    jsonData[0].transcriptions[key].actions.escalation.status =
                        "N";
                }
                if (matches[0].toString().includes("help:Y")) {
                    jsonData[0].transcriptions[key].ask.help.status = "Y";
                }
                if (matches[0].toString().includes("help:N")) {
                    jsonData[0].transcriptions[key].ask.help.status = "N";
                }
                if (matches[0].toString().includes("questions:Y")) {
                    jsonData[0].transcriptions[key].ask.questions.status = "Y";
                }
                if (matches[0].toString().includes("questions:N")) {
                    jsonData[0].transcriptions[key].ask.questions.status = "N";
                }
                if (matches[0].toString().includes("sentiment:POS")) {
                    jsonData[0].transcriptions[key].sentiment = "positive";
                }
                if (matches[0].toString().includes("sentiment:NEU")) {
                    jsonData[0].transcriptions[key].sentiment = "neutral";
                }
                if (matches[0].toString().includes("sentiment:NEG")) {
                    jsonData[0].transcriptions[key].sentiment = "negative";
                }
            }
        }
    };

    pushtoNERJson = (idVal, label) => {
        var text;
        if ("correctedText" in jsonData[0].transcriptions[idVal]) {
            text = String(jsonData[0].transcriptions[idVal].correctedText);
        } else {
            text = String(jsonData[0].transcriptions[idVal].text);
        }
        NERchangesJsonData.push({
            id: idVal,
            text: text.replace(/\{(.*?)\}/g, ""),
            label: label,
        });
    };

    render() {
        return (
            <div>
                <p>
                    {document
                        .getElementById("globalSaveButton")
                        .addEventListener("click", () => {
                            this.handleSubmit();
                        })}

                    {document
                        .getElementById("globalEditButton")
                        .addEventListener("click", () => {
                            this.setState({ editing: true });
                        })}
                </p>
                <TranscriptDash
                    meeting={
                        this.state.new_transcription.length === 0
                            ? this.state.meeting
                            : this.state.new_transcription
                    }
                    editing={this.state.editing}
                    btnName={this.props.btnName ? this.props.btnName : ""}
                />
            </div>
        );
    }
}

class TranscriptDash extends React.Component {
    render() {
        let trans = this.props.meeting.map((trans1) => (
            <Transcript
                key={trans1.key}
                transcript={trans1.text}
                time={trans1.start_time}
                ids={trans1.speakerId}
                editing={this.props.editing}
            />
        ));
        let filterName = "";
        filterName = this.props.btnName;
        if (filterName.includes("actionBtn")) {
            trans = this.props.meeting.map((trans1, id) => {
                if (trans1.actions.deadline.status === "Y") {
                    return (
                        <Transcript
                            key={trans1.key}
                            transcript={trans1.text}
                            time={trans1.start_time}
                            ids={trans1.speakerId}
                            editing={this.props.editing}
                        />
                    );
                }
            });
        }
        if (filterName.includes("updatesBtn")) {
            trans = this.props.meeting.map((trans1, id) => {
                if (trans1.actions.deadline.status === "Y") {
                    return (
                        <Transcript
                            key={trans1.key}
                            transcript={trans1.text}
                            time={trans1.start_time}
                            ids={trans1.speakerId}
                            editing={this.props.editing}
                        />
                    );
                }
            });
        }

        if (filterName.includes("escalationsBtn")) {
            trans = this.props.meeting.map((trans1, id) => {
                if (trans1.actions.escalation.status === "Y") {
                    return (
                        <Transcript
                            key={trans1.key}
                            transcript={trans1.text}
                            time={trans1.start_time}
                            ids={trans1.speakerId}
                            editing={this.props.editing}
                        />
                    );
                }
            });
        }

        if (filterName.includes("asksBtn")) {
            trans = this.props.meeting.map((trans1, id) => {
                if (
                    trans1.ask.help.status === "Y" ||
                    trans1.ask.questions.status === "Y"
                ) {
                    return (
                        <Transcript
                            key={trans1.key}
                            transcript={trans1.text}
                            time={trans1.start_time}
                            ids={trans1.speakerId}
                            editing={this.props.editing}
                        />
                    );
                }
            });
        }

        if (filterName.includes("recommendationBtn")) {
            trans = this.props.meeting.map((trans1, id) => {
                if (
                    trans1.entities.name.length !== 0 ||
                    trans1.entities.tools.length !== 0 ||
                    trans1.entities.technology.length !== 0 ||
                    trans1.entities.anthemTools.length !== 0
                ) {
                    return (
                        <Transcript
                            key={trans1.key}
                            transcript={trans1.text}
                            time={trans1.start_time}
                            ids={trans1.speakerId}
                            editing={this.props.editing}
                        />
                    );
                }
            });
        }

        if (filterName.includes("callOutBtn")) {
            trans = this.props.meeting.map((trans1, id) => {
                if (trans1.entities.name.length !== 0) {
                    return (
                        <Transcript
                            key={trans1.key}
                            transcript={trans1.text}
                            time={trans1.start_time}
                            ids={trans1.speakerId}
                            editing={this.props.editing}
                        />
                    );
                }
            });
        }

        if (filterName.includes("negativeBtn")) {
            trans = this.props.meeting.map((trans1, id) => {
                if (trans1.sentiment === "negative") {
                    return (
                        <Transcript
                            key={trans1.key}
                            transcript={trans1.text}
                            time={trans1.start_time}
                            ids={trans1.speakerId}
                            editing={this.props.editing}
                        />
                    );
                }
            });
        }

        if (filterName.includes("positiveBtn")) {
            trans = this.props.meeting.map((trans1, id) => {
                if (trans1.sentiment === "positive") {
                    return (
                        <Transcript
                            key={trans1.key}
                            transcript={trans1.text}
                            time={trans1.start_time}
                            ids={trans1.speakerId}
                            editing={this.props.editing}
                        />
                    );
                }
            });
        }

        if (filterName.includes("neutralBtn")) {
            trans = this.props.meeting.map((trans1, id) => {
                if (trans1.sentiment === "neutral") {
                    return (
                        <Transcript
                            key={trans1.key}
                            transcript={trans1.text}
                            time={trans1.start_time}
                            ids={trans1.speakerId}
                            editing={this.props.editing}
                        />
                    );
                }
            });
        }

        return <div id="transcript2">{trans}</div>;
    }
}

class Transcript extends React.Component {
    state = {
        editing: false,
        transcript: this.props.transcript,
        ids: this.props.ids,
        time: 1,
        buttons: [
            {
                label: "Mark as Deadline",
                onClick: () => this.handleRightClickNERs("deadline", "Y"),
            },
            {
                label: "Unmark as Deadline",
                onClick: () => this.handleRightClickNERs("deadline", "N"),
            },
            {
                label: "Mark as Escalation",
                onClick: () => this.handleRightClickNERs("escalation", "Y"),
            },
            {
                label: "Unmark as Escalation",
                onClick: () => this.handleRightClickNERs("escalation", "N"),
            },
            {
                label: "Mark as Help",
                onClick: () => this.handleRightClickNERs("help", "Y"),
            },
            {
                label: "Unmark as Help",
                onClick: () => this.handleRightClickNERs("help", "N"),
            },
            {
                label: "Mark as Questions",
                onClick: () => this.handleRightClickNERs("questions", "Y"),
            },
            {
                label: "Unmark as Questions",
                onClick: () => this.handleRightClickNERs("questions", "N"),
            },
            {
                label: "Mark Positive Sentiment",
                onClick: () => this.handleRightClickNERs("sentiment", "POS"),
            },
            {
                label: "Mark Neutral Sentiment",
                onClick: () => this.handleRightClickNERs("sentiment", "NEU"),
            },
            {
                label: "Mark Negative Sentiment",
                onClick: () => this.handleRightClickNERs("sentiment", "NEG"),
            },
            {
                label: "Mark as Name",
                onClick: () =>
                    this.handleRightClickNERs(
                        "name",
                        window.getSelection().toString()
                    ),
            },
            {
                label: "Mark as Tool",
                onClick: () =>
                    this.handleRightClickNERs(
                        "tool",
                        window.getSelection().toString()
                    ),
            },
            {
                label: "Mark as Anthem Tool",
                onClick: () =>
                    this.handleRightClickNERs(
                        "anthemTool",
                        window.getSelection().toString()
                    ),
            },
            {
                label: "Mark as Technology",
                onClick: () =>
                    this.handleRightClickNERs(
                        "technology",
                        window.getSelection().toString()
                    ),
            },
        ],
        buttonsRules: [
            {
                label: "Extra words by machine",
                onClick: () => this.handleRightClickRules("[]", "[/]"),
            },
            {
                label: "Missed word by machine",
                onClick: () => this.handleRightClickRules("[-]", "[/-]"),
            },
            {
                label: "Speaker Separation",
                onClick: () => this.handleRightClickRules("[sp]", "[sp]"),
            },
            {
                label: "Word/sentence not clear",
                onClick: () => this.handleRightClickRules("[**]", ""),
            },
            {
                label: "Machine skipped to transcribe the sentence",
                onClick: () => this.handleRightClickRules("[$]", "[/$]"),
            },
            {
                label: "Correct word/Sentence",
                onClick: () => this.handleRightClickRules("[#]", "[/#]"),
            },
            {
                label: "Misinterpreted word",
                onClick: () => this.handleRightClickRules("< >", "</>"),
            },
            {
                label: "Misinterpreted sentence",
                onClick: () => this.handleRightClickRules("()", "(/)"),
            },
            {
                label: "Personal Information",
                onClick: () => this.handleRightClickRules("[PI]", "[/PI]"),
            },
            {
                label: "Intelligent guess",
                onClick: () => this.handleRightClickRules("[??]", "[/??]"),
            },
            {
                label: "Guess of misinterpreted word",
                onClick: () => this.handleRightClickRules("(*)", "(/*)"),
            },
            {
                label: "Human laughter/noise",
                onClick: () => this.handleRightClickRules("[%]", ""),
            },
            {
                label: "Similar way pronounced",
                onClick: () => this.handleRightClickRules("[@]", "[/@]"),
            },
            {
                label: "Word recorded wrong w.r.t spelling",
                onClick: () => this.handleRightClickRules("[sc]", "[/sc]"),
            },
            //{ label: 'check', onClick: (e) => alert(`Rule 10`) },
        ],
        value: [],
        tag: "NAME",
    };

    onInputchange = (change) => {
        this.setState({
            transcript: change.target.value,
        });
        jsonData = JSON.parse(sessionStorage.getItem("updatedData"))
            ? JSON.parse(sessionStorage.getItem("updatedData"))
            : jsonData;
        //console.log(jsonData);
        jsonData[0].transcriptions[this.state.ids - 1].text =
            change.target.value;
        //console.log(jsonData);
        changesMadeNER.add(this.state.ids - 1);
        if (changesMade[changesMade.length - 1] !== this.state.ids - 1) {
            changesMade.push(this.state.ids - 1);
        }
        sessionStorage.setItem("updatedData", JSON.stringify(jsonData));
        document.getElementById("transcript-" + String(this.state.ids)).value =
            change.target.value;
    };

    handleRightClickRules = (textBefore, textAfter) => {
        let textVal = this.refs.textareaSelect;
        let cursorStart = textVal.selectionStart;
        let cursorEnd = textVal.selectionEnd;
        //console.log(cursorStart, cursorEnd);
        if (cursorStart === cursorEnd && textAfter !== "") {
            return;
        }

        //console.log(this.state.transcript);
        //console.log(textBefore, textAfter);

        jsonData = JSON.parse(sessionStorage.getItem("updatedData"))
            ? JSON.parse(sessionStorage.getItem("updatedData"))
            : jsonData;
        this.setState(
            {
                transcript:
                    jsonData[0].transcriptions[this.state.ids - 1].text.slice(
                        0,
                        cursorStart
                    ) +
                    textBefore +
                    jsonData[0].transcriptions[this.state.ids - 1].text.slice(
                        cursorStart,
                        cursorEnd
                    ) +
                    textAfter +
                    jsonData[0].transcriptions[this.state.ids - 1].text.slice(
                        cursorEnd,
                        jsonData[0].transcriptions[this.state.ids - 1].text
                            .length
                    ),
            },
            () => {
                //console.log(this.state.transcript, 'hereeeeeeeee');
                jsonData[0].transcriptions[this.state.ids - 1].text =
                    this.state.transcript;
                changesMadeNER.add(this.state.ids - 1);
                if (
                    changesMade[changesMade.length - 1] !==
                    this.state.ids - 1
                ) {
                    changesMade.push(this.state.ids - 1);
                }
                sessionStorage.setItem("updatedData", JSON.stringify(jsonData));
                document.getElementById(
                    "textarea-" + String(this.state.ids)
                ).value = this.state.transcript;
                // console.log(
                //   document.getElementById('textarea-' + String(this.state.ids)).value
                // );
            }
        );
    };

    handleRightClickNERs = (labelName, labelValue) => {
        if (labelValue) {
            jsonData = JSON.parse(sessionStorage.getItem("updatedData"))
                ? JSON.parse(sessionStorage.getItem("updatedData"))
                : jsonData;
            var trans = jsonData[0].transcriptions[this.state.ids - 1].text;
            if (/\{(.*?)\}/.test(trans)) {
                const matches = trans.match(/\{(.*?)\}/g);
                if (matches[0].toString().includes(labelName)) {
                    if (
                        labelName == "name" ||
                        labelName == "tool" ||
                        labelName == "anthemTool" ||
                        labelName == "technology"
                    ) {
                        var labels = trans.replace(
                            matches[0],
                            matches[0].toString().slice(0, -1) +
                                ", " +
                                labelName +
                                ":" +
                                labelValue +
                                "}"
                        );
                        this.setState(
                            {
                                transcript: labels,
                            },
                            () => {
                                //console.log(this.state.transcript);
                                jsonData[0].transcriptions[
                                    this.state.ids - 1
                                ].text = this.state.transcript;
                                changesMadeNER.add(this.state.ids - 1);
                                if (
                                    changesMade[changesMade.length - 1] !==
                                    this.state.ids - 1
                                ) {
                                    changesMade.push(this.state.ids - 1);
                                }
                                sessionStorage.setItem(
                                    "updatedData",
                                    JSON.stringify(jsonData)
                                );
                            }
                        );
                    } else {
                        var regex = new RegExp(
                            "(" + labelName + ")(:)(.*?)[},]",
                            "g"
                        );
                        const matches1 = matches[0].toString().match(regex);
                        var temp = matches1[0]
                            .toString()
                            .slice(0, matches1[0].toString().length - 1);
                        var temp1 = matches[0]
                            .toString()
                            .replace(temp, labelName + ":" + labelValue);
                        var labels = trans.replace(matches[0], temp1);
                        this.setState(
                            {
                                transcript: labels,
                            },
                            () => {
                                //console.log(this.state.transcript);
                                jsonData[0].transcriptions[
                                    this.state.ids - 1
                                ].text = this.state.transcript;
                                changesMadeNER.add(this.state.ids - 1);
                                if (
                                    changesMade[changesMade.length - 1] !==
                                    this.state.ids - 1
                                ) {
                                    changesMade.push(this.state.ids - 1);
                                }
                                sessionStorage.setItem(
                                    "updatedData",
                                    JSON.stringify(jsonData)
                                );
                            }
                        );
                    }
                } else {
                    var labels = trans.replace(
                        matches[0],
                        matches[0].toString().slice(0, -1) +
                            ", " +
                            labelName +
                            ":" +
                            labelValue +
                            "}"
                    );
                    this.setState(
                        {
                            transcript: labels,
                        },
                        () => {
                            //console.log(this.state.transcript);
                            jsonData[0].transcriptions[
                                this.state.ids - 1
                            ].text = this.state.transcript;
                            changesMadeNER.add(this.state.ids - 1);
                            if (
                                changesMade[changesMade.length - 1] !==
                                this.state.ids - 1
                            ) {
                                changesMade.push(this.state.ids - 1);
                            }
                            sessionStorage.setItem(
                                "updatedData",
                                JSON.stringify(jsonData)
                            );
                        }
                    );
                }
            } else {
                this.setState({
                    transcript:
                        "{" +
                        labelName +
                        ":" +
                        labelValue +
                        "}" +
                        jsonData[0].transcriptions[this.state.ids - 1].text,
                });
                //console.log(this.state.transcript);
                jsonData[0].transcriptions[this.state.ids - 1].text =
                    this.state.transcript;
                changesMadeNER.add(this.state.ids - 1);
                if (
                    changesMade[changesMade.length - 1] !==
                    this.state.ids - 1
                ) {
                    changesMade.push(this.state.ids - 1);
                }
                sessionStorage.setItem("updatedData", JSON.stringify(jsonData));
            }
        }
    };

    handleRightClickNERscopy = (labelName, labelValue) => {
        if (labelValue) {
            jsonData = JSON.parse(sessionStorage.getItem("updatedData"))
                ? JSON.parse(sessionStorage.getItem("updatedData"))
                : jsonData;
            var trans = jsonData[0].transcriptions[this.state.ids - 1].text;
            if (/\{(.*?)\}/.test(trans)) {
                const matches = trans.match(/\{(.*?)\}/g);
                if (matches[0].toString().includes(labelName)) {
                    var regex = new RegExp(
                        "(" + labelName + ")(:)(.*?)[};]",
                        "g"
                    );
                    const matches1 = matches[0].toString().match(regex);
                    var temp = matches1[0]
                        .toString()
                        .slice(0, matches1[0].toString().length - 1);
                    var temp1 = matches[0]
                        .toString()
                        .replace(temp, labelName + ":" + labelValue);
                    if (
                        labelName == "name" ||
                        labelName == "tool" ||
                        labelName == "anthemTool" ||
                        labelName == "technology"
                    ) {
                        temp1 = matches[0]
                            .toString()
                            .replace(temp, temp + ", " + labelValue);
                    }

                    var labels = trans.replace(matches[0], temp1);
                    this.setState(
                        {
                            transcript: labels,
                        },
                        () => {
                            //console.log(this.state.transcript);
                            jsonData[0].transcriptions[
                                this.state.ids - 1
                            ].text = this.state.transcript;
                            changesMadeNER.add(this.state.ids - 1);
                            if (
                                changesMade[changesMade.length - 1] !==
                                this.state.ids - 1
                            ) {
                                changesMade.push(this.state.ids - 1);
                            }
                            sessionStorage.setItem(
                                "updatedData",
                                JSON.stringify(jsonData)
                            );
                        }
                    );
                } else {
                    var labels = trans.replace(
                        matches[0],
                        matches[0].toString().slice(0, -1) +
                            "; " +
                            labelName +
                            ":" +
                            labelValue +
                            "}"
                    );
                    this.setState(
                        {
                            transcript: labels,
                        },
                        () => {
                            //console.log(this.state.transcript);
                            jsonData[0].transcriptions[
                                this.state.ids - 1
                            ].text = this.state.transcript;
                            changesMadeNER.add(this.state.ids - 1);
                            if (
                                changesMade[changesMade.length - 1] !==
                                this.state.ids - 1
                            ) {
                                changesMade.push(this.state.ids - 1);
                            }
                            sessionStorage.setItem(
                                "updatedData",
                                JSON.stringify(jsonData)
                            );
                        }
                    );
                }
            } else {
                this.setState(
                    {
                        transcript:
                            "{" +
                            labelName +
                            ":" +
                            labelValue +
                            "}" +
                            jsonData[0].transcriptions[this.state.ids - 1].text,
                    },
                    () => {
                        //console.log(this.state.transcript);
                        jsonData[0].transcriptions[this.state.ids - 1].text =
                            this.state.transcript;
                        changesMadeNER.add(this.state.ids - 1);
                        if (
                            changesMade[changesMade.length - 1] !==
                            this.state.ids - 1
                        ) {
                            changesMade.push(this.state.ids - 1);
                        }
                        sessionStorage.setItem(
                            "updatedData",
                            JSON.stringify(jsonData)
                        );
                    }
                );
            }
        }
    };

    handleChange = (value) => {
        this.setState({ value });
    };

    handleTagChange = (e) => {
        this.setState({ tag: e.target.value });
    };

    RandomNumber = Math.floor((Math.random() * 100) / 10) + 1;

    date = new Date(this.props.time * 100);
    hours = this.date.getUTCHours();
    minutes = this.date.getUTCMinutes();
    seconds = this.date.getSeconds();

    timeString =
        this.hours.toString().padStart(2, "0") +
        ":" +
        this.minutes.toString().padStart(2, "0") +
        ":" +
        this.seconds.toString().padStart(2, "0");

    render() {
        return (
            <div>
                <div
                    className="transcript-individual"
                    id={`transcript-${this.state.ids}`}
                >
                    <header class="sc-hEsumM owEmt">
                        <span>
                            <img src="img/a1.svg" alt="speaker" />
                        </span>
                        <span tabindex="0" class="">
                            User {this.RandomNumber}
                        </span>
                        <span className="dot"></span>
                        <span>{this.timeString}</span>
                    </header>
                    <p class="sc-ktHwxA hmbalP" style={divStyle}>
                        <span id="p-0" data-index="0" class="sc-cIShpX hoCubg">
                            <span
                                class="sc-jnlKLf gCJFYe cap-sent-0 cap-time-0.09--7.14 heightfindersent0"
                                data-index="0"
                                data-key="0"
                            >
                                {this.props.editing ? (
                                    <ContextMenu
                                        buttons={this.state.buttonsRules}
                                    >
                                        <textarea
                                            className="transcript-textarea"
                                            ref="textareaSelect"
                                            id={`textarea-${this.state.ids}`}
                                            defaultValue={this.state.transcript}
                                            onChange={this.onInputchange}
                                            onMouseUpCapture={this.selectedText}
                                        />
                                    </ContextMenu>
                                ) : (
                                    <ContextMenu buttons={this.state.buttons}>
                                        {/* <HighlightWord texttohighlight={this.state.transcript} errorToHighlight={jsonData[0].transcriptions[this.state.ids - 1].errorDetails} /> */}
                                        {/* <Card> */}
                                        <select
                                            onChange={this.handleTagChange}
                                            value={this.state.tag}
                                        >
                                            <option value="NAME">NAME</option>
                                            <option value="TOOL">TOOL</option>
                                            <option value="ANTHEMTOOL">
                                                ANTHEM TOOL
                                            </option>
                                            <option value="TECHNOLOGY">
                                                TECHNOLOGY
                                            </option>
                                        </select>

                                        <TextAnnotator
                                            style={{
                                                maxWidth: 500,
                                                lineHeight: 1.5,
                                            }}
                                            content={this.state.transcript}
                                            value={this.state.value}
                                            onChange={this.handleChange}
                                            getSpan={(span) => ({
                                                ...span,
                                                tag: this.state.tag,
                                                color: TAG_COLORS[
                                                    this.state.tag
                                                ],
                                            })}
                                            renderMark={(props) => (
                                                <mark
                                                    key={props.key}
                                                    onClick={() =>
                                                        props.onClick({
                                                            start: props.start,
                                                            end: props.end,
                                                            text: props.text,
                                                            tag: props.tag,
                                                            color: TAG_COLORS[
                                                                props.tag
                                                            ],
                                                        })
                                                    }
                                                    style={{
                                                        padding: ".2em .3em",
                                                        margin: "0 .25em",
                                                        lineHeight: "1",
                                                        display: "inline-block",
                                                        borderRadius: ".25em",
                                                        background:
                                                            TAG_COLORS[
                                                                props.tag
                                                            ],
                                                    }}
                                                >
                                                    {props.content}{" "}
                                                    <span
                                                        style={{
                                                            boxSizing:
                                                                "border-box",
                                                            content:
                                                                "attr(data-entity)",
                                                            fontSize: "2.55em",
                                                            lineHeight: "1",
                                                            padding:
                                                                ".35em .35em",
                                                            borderRadius:
                                                                ".35em",
                                                            textTransform:
                                                                "lowercase",
                                                            display:
                                                                "inline-block",
                                                            verticalAlign:
                                                                "middle",
                                                            margin: "0 0 .15rem .5rem",
                                                            background: "#fff",
                                                            fontWeight: "700",
                                                        }}
                                                    >
                                                        {" "}
                                                        {props.tag}
                                                    </span>
                                                </mark>
                                            )}
                                        />
                                        {/* </Card> */}
                                        <Card>
                                            <h4>Current Value</h4>
                                            <pre>
                                                {JSON.stringify(
                                                    this.state.value,
                                                    null,
                                                    2
                                                )}
                                            </pre>
                                        </Card>
                                    </ContextMenu>
                                )}
                            </span>
                        </span>
                    </p>
                </div>
            </div>
        );
    }
}

class ContextMenu extends React.Component {
    static defaultProps = {
        buttons: [],
    };

    constructor() {
        super();

        this.state = {
            open: false,
        };
    }

    componentDidMount() {
        document.addEventListener("click", this.handleClickOutside);
        document.addEventListener("contextmenu", this.handleRightClickOutside);
    }

    handleClickOutside = (e) => {
        if (!this.state.open) {
            return;
        } else {
            this.setState({
                open: false,
            });
        }

        // const root = ReactDOM.findDOMNode(this.div);
        // const context = ReactDOM.findDOMNode(this.context);
        // const isInRow = (!root.contains(e.target) || root.contains(e.target));
        // const isInContext = !context.contains(e.target);

        // if (isInRow && isInContext) {
        //   this.setState({
        //     open: false
        //   });
        // }
    };

    handleRightClickOutside = (e) => {
        if (!this.state.open) {
            return;
        }

        const root = ReactDOM.findDOMNode(this.div);
        const isInRow = !root.contains(e.target);

        if (isInRow) {
            this.setState({
                open: false,
            });
        }
    };

    handleRightClick = (e) => {
        e.preventDefault();
        this.setState({
            open: true,
            top: window.scrollY + e.nativeEvent.clientY,
            left: e.nativeEvent.clientX,
        });
    };

    render() {
        return (
            <div
                onContextMenu={this.handleRightClick}
                ref={(node) => (this.div = node)}
            >
                {this.props.children}

                {!this.state.open ? null : (
                    <div
                        className="context"
                        ref={(div) => (this.context = div)}
                        style={{ top: this.state.top, left: this.state.left }}
                    >
                        <ul>
                            {
                                // button - name, onClick, label
                                this.props.buttons.length > 0 &&
                                    this.props.buttons.map((button) => {
                                        return (
                                            <li key={button.label}>
                                                <a
                                                    href="#"
                                                    onClick={button.onClick}
                                                >
                                                    {button.label}
                                                </a>
                                                {this.handleRightClick}
                                            </li>
                                        );
                                    })
                            }
                        </ul>
                    </div>
                )}
            </div>
        );
    }
}

export default TranscriptDashboard;
