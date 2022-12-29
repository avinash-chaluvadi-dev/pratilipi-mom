import { MenuItem as MenuItemTag } from "react-contextmenu";
import React from "react";
import { useSelector } from "react-redux";
import customStyles from "screens/FeedBackLoop/components/DetailedView/styles";

const RightClickRules = (props) => {
    let { startIndex, endIndex, openContext, item, selectedIdx, textObjects } =
        props;

    const detailCls = customStyles();

    const { momStore } = useSelector((state) => state.momReducer);
    let MoMStoreData = momStore?.concatenated_view;

    const handleRightClickRules = (textBefore, textAfter, index) => {
        if (openContext) {
            let cursorStart = startIndex;
            let cursorEnd = endIndex;
            if (cursorStart === cursorEnd && textAfter !== "") {
                return;
            }
            if (
                MoMStoreData[index] &&
                MoMStoreData[index].transcript &&
                MoMStoreData[index].transcript.startsWith("[SOT]")
            ) {
                MoMStoreData[index].transcript =
                    MoMStoreData[index].transcript.slice(0, cursorStart) +
                    textBefore +
                    MoMStoreData[index].transcript.slice(
                        cursorStart,
                        cursorEnd
                    ) +
                    textAfter +
                    MoMStoreData[index].transcript.slice(
                        cursorEnd,
                        MoMStoreData[index].transcript.length
                    );
                textObjects[index] =
                    textObjects[index]?.slice(0, cursorStart) +
                    textBefore +
                    textObjects[index]?.slice(cursorStart, cursorEnd) +
                    textAfter +
                    textObjects[index]?.slice(
                        cursorEnd,
                        textObjects[index]?.length
                    );
            } else {
                MoMStoreData[index].transcript =
                    "[SOT]" +
                    MoMStoreData[index].transcript.slice(0, cursorStart) +
                    textBefore +
                    MoMStoreData[index].transcript.slice(
                        cursorStart,
                        cursorEnd
                    ) +
                    textAfter +
                    MoMStoreData[index].transcript.slice(
                        cursorEnd,
                        MoMStoreData[index].transcript.length
                    ) +
                    "[EOT]";

                textObjects[index] =
                    "[SOT]" +
                    textObjects[index]?.slice(0, cursorStart) +
                    textBefore +
                    textObjects[index]?.slice(cursorStart, cursorEnd) +
                    textAfter +
                    textObjects[index]?.slice(
                        cursorEnd,
                        textObjects[index]?.length
                    ) +
                    "[EOT]";
            }
            props.updateDataEvent(textObjects);
        }
    };

    return (
        <MenuItemTag
            onClick={() =>
                handleRightClickRules(item.start, item.end, selectedIdx)
            }
            className={detailCls.contextmenulist}
            key={Math.random()}
        >
            {item.label}
        </MenuItemTag>
    );
};

export default RightClickRules;
