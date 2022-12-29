import { SummarycolorCheckEntity } from "screens/FeedBackLoop/masterData";

export const isEmptyObject = (obj) => {
    return JSON.stringify(obj) === "{}";
};

export const isArray = (object) => {
    if (object) {
        return object.constructor === Array;
    }
};

export const prepareArr = (obj) => {
    let arr = [];
    if (!isArray(obj)) {
        //!obj?.length > 0) {
        obj &&
            !isEmptyObject(obj) &&
            Object.keys(obj)?.length > 0 &&
            Object.keys(obj).forEach((key) => {
                if (isNaN(key)) {
                    arr.push(`${key}`);
                }
            });
    } else {
        arr = obj;
    }
    return arr;
};

export const getStartEndStart = (string, words, type) => {
    const markers = [];
    words?.forEach((word, idx) => {
        const index = string.indexOf(word);
        if (index !== -1) {
            const endIndex = index + word.length;
            markers.push({ start: index, end: endIndex, word: type[idx] });
        }
    });
    return markers;
};

export const markSelections = (text, markers) => {
    const sortedMarkers = [...markers].sort((m1, m2) => m1.start - m2.start);
    let markedText = "";
    let characterPointer = 0;
    sortedMarkers.forEach(({ start, end, word }) => {
        markedText += text.substring(characterPointer, start);
        markedText += `<mark title=${word} style="color: ${SummarycolorCheckEntity(
            word
        )}" >`;
        markedText += text.substring(start, end);
        markedText += "</mark>";
        characterPointer = end;
    });
    markedText += text.substring(characterPointer);
    return markedText;
};

export const loadAssignedValues = (momStore) => {
    let finalList = momStore?.concatenated_view?.filter(
        (v, i, a) => a.findIndex((t) => t.speaker_id === v.speaker_id) === i
    );
    finalList = finalList.map((item) => {
        return {
            ...item,
            // value: 'user 01',
            // label: 'user 01',
            value: momStore["map_username"][item.speaker_id],
            label: momStore["map_username"][item.speaker_id],
            assign_to: momStore["map_username"][item.speaker_id],
        };
    });
    let data = momStore["ExternalParticipants"]
        ? momStore["ExternalParticipants"]
        : [];
    // let getVal = sessionStorage.getItem("saveExternalUsers");
    // let saveExternalUsersVal = getVal ? JSON.parse(getVal) : [];
    //console.log('===padapada===', saveExternalUsersVal);
    momStore["AssignTo"] = [...finalList, ...data];
    return momStore;
};
