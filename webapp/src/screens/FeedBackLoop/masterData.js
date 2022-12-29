import actionsIcon from "static/images/actions.svg";
import EscalationsIcon from "static/images/escalations.svg";
import DeadLinesIcon from "static/images/deadlines.svg";
import RecommendationsIcon from "static/images/recommendations.svg";
import AppreciationsIcon from "static/images/appreciations.svg";
import DateIcon from "static/images/Date.svg";
import EventIcon from "static/images/Event.svg";
import NameIcon from "static/images/Name.svg";
import AnthemToolIcon from "static/images/AnthemTool.svg";
import ToolslIcon from "static/images/Tools.svg";
import TechnicalPlatformIcon from "static/images/TechnicalPlatform.svg";
import ScrumIcon from "static/images/Scrum.svg";
import StatusIcon from "static/images/Status.svg";
import positiveIcon from "static/images/positive.svg";
import neutralIcon from "static/images/neutral.svg";
import negativeIcon from "static/images/negative.svg";
import useStyles from "screens/FeedBackLoop/styles";

export const FilterOptions = () => {
    return [
        { value: "Labels", label: "Labels" },
        { value: "Entities", label: "Entities" },
        { value: "Sentiments", label: "Sentiments" },
        { value: "Participants", label: "Participants" },
    ];
};
export const MakerdataOptions = () => {
    return [
        { value: "Mentoring & Engagement", label: "Mentoring & Engagement" },
        { value: "Proactiveness", label: "Proactiveness" },
        { value: "Action Plan Tracking", label: "Action Plan Tracking" },
        { value: "Collabaration", label: "Collabaration" },
    ];
};
export const LabelDataOptions = () => {
    const classes = useStyles();
    return [
        {
            value: "Action",
            label: (
                <div className={classes.textOverflow}>
                    <img src={actionsIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Actions</span>
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#2F71D8" }}>Actions</span>
                </div>
            ),
        },
        {
            value: "Action with Deadline",
            label: (
                <div className={classes.textOverflow}>
                    <img
                        src={DeadLinesIcon}
                        height="10px"
                        width="10px"
                        alt=""
                    />{" "}
                    <span style={{ padding: "8px" }}>Action with Deadline</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#E05C5C" }}>
                        Action with Deadline
                    </span>{" "}
                </div>
            ),
        },
        {
            value: "Announcement",
            label: (
                <div className={classes.textOverflow}>
                    <img
                        src={RecommendationsIcon}
                        height="10px"
                        width="10px"
                        alt=""
                    />{" "}
                    <span style={{ padding: "8px" }}>Announcement</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#6BD052" }}>Announcement</span>{" "}
                </div>
            ),
        },
        {
            value: "Appreciation",
            label: (
                <div className={classes.textOverflow}>
                    <img
                        src={AppreciationsIcon}
                        height="10px"
                        width="10px"
                        alt=""
                    />{" "}
                    <span style={{ padding: "8px" }}>Appreciations</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#53A0FA" }}>Appreciations</span>{" "}
                </div>
            ),
        },
        {
            value: "Escalation",
            label: (
                <div className={classes.textOverflow}>
                    <img
                        src={EscalationsIcon}
                        height="10px"
                        width="10px"
                        alt=""
                    />{" "}
                    <span style={{ padding: "8px" }}>Escalations</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#889943" }}>Escalations</span>{" "}
                </div>
            ),
        },
    ];
};
export const EntitiesDataOptions = () => {
    const classes = useStyles();

    return [
        {
            value: "Date",
            label: (
                <div className={classes.textOverflow}>
                    <img src={DateIcon} height="10px" width="10px" alt="" />{" "}
                    <span
                        style={{ padding: "8px" }}
                        onClick={
                            () => {}
                            // handleRightClickNERs('date', 'Y', selectedIdx, 'Mark as Date')
                        }
                    >
                        Date
                    </span>
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span
                        style={{ color: "#C340A6" }}
                        onClick={
                            () => {}
                            // handleRightClickNERs('date', 'Y', selectedIdx, 'Mark as Date')
                        }
                    >
                        Date
                    </span>
                </div>
            ),
        },
        {
            value: "Event",
            label: (
                <div
                    className={classes.textOverflow}
                    onClick={
                        () => {}
                        //   handleRightClickNERs('event', 'Y', selectedIdx, 'Mark as Event')
                    }
                >
                    <img src={EventIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Event</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div
                    className={classes.textOverflow}
                    onClick={
                        () => {}
                        //   handleRightClickNERs('event', 'Y', selectedIdx, 'Mark as Event')
                    }
                >
                    <span style={{ color: "#40C3AC" }}>Event</span>{" "}
                </div>
            ),
        },
        {
            value: "Name",
            label: (
                <div
                    className={classes.textOverflow}
                    onClick={
                        () => {}
                        //   handleRightClickNERs('name', 'Y', selectedIdx, 'Mark as Name')
                    }
                >
                    <img src={NameIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Name</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div
                    className={classes.textOverflow}
                    onClick={
                        () => {}
                        //   handleRightClickNERs('name', 'Y', selectedIdx, 'Mark as Name')
                    }
                >
                    <span style={{ color: "#994B20" }}>Name</span>{" "}
                </div>
            ),
        },
        {
            value: "Anthemtool",
            label: (
                <div
                    className={classes.textOverflow}
                    onClick={
                        () => {}
                        //   handleRightClickNERs('anthemTool', 'Y', selectedIdx, 'Mark as AnthemTool')
                    }
                >
                    <img
                        src={AnthemToolIcon}
                        height="10px"
                        width="10px"
                        alt=""
                    />{" "}
                    <span style={{ padding: "8px" }}>Anthem tool</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div
                    className={classes.textOverflow}
                    onClick={
                        () => {}
                        //   handleRightClickNERs('anthemTool', 'Y', selectedIdx, 'Mark as AnthemTool')
                    }
                >
                    <span style={{ color: "#DF3131" }}>Anthem tool</span>{" "}
                </div>
            ),
        },
        {
            value: "Tool",
            label: (
                <div
                    className={classes.textOverflow}
                    onClick={
                        () => {}
                        //   handleRightClickNERs('tool', 'Y', selectedIdx, 'Mark as Tool')
                    }
                >
                    <img src={ToolslIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Tool</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div
                    className={classes.textOverflow}
                    onClick={
                        () => {}
                        //   handleRightClickNERs('tool', 'Y', selectedIdx, 'Mark as Tool')
                    }
                >
                    <span style={{ color: "#256D13" }}>Tool</span>{" "}
                </div>
            ),
        },
        {
            value: "Technicalplatform",
            label: (
                <div className={classes.textOverflow}>
                    <img
                        src={TechnicalPlatformIcon}
                        height="10px"
                        width="10px"
                        alt=""
                    />{" "}
                    <span style={{ padding: "8px" }}>Technical platform</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#552099" }}>Technical platform</span>{" "}
                </div>
            ),
        },
        {
            value: "Scrum",
            label: (
                <div className={classes.textOverflow}>
                    <img src={ScrumIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Scrum</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#D73AAB" }}>Scrum</span>{" "}
                </div>
            ),
        },
        {
            value: "Status",
            label: (
                <div className={classes.textOverflow}>
                    <img src={StatusIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Status</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#53A0FA" }}>Status</span>{" "}
                </div>
            ),
        },
        {
            value: "Organization",
            label: (
                <div className={classes.textOverflow}>
                    <img src={StatusIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Organization</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#53A0FA" }}>Organization</span>{" "}
                </div>
            ),
        },
        {
            value: "Technologies",
            label: (
                <div className={classes.textOverflow}>
                    <img src={StatusIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Technology</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#53A0FA" }}>Technology</span>{" "}
                </div>
            ),
        },
        {
            value: "Team name",
            label: (
                <div className={classes.textOverflow}>
                    <img src={StatusIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Team name</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#53A0FA" }}>Team name</span>{" "}
                </div>
            ),
        },
    ];
};
export const SentimentDataOptions = () => {
    const classes = useStyles();

    return [
        {
            value: "positive",
            label: (
                <div className={classes.textOverflow}>
                    <img src={positiveIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Positive</span>
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#34B53A" }}>Positive</span>
                </div>
            ),
        },
        {
            value: "neutral",
            label: (
                <div className={classes.textOverflow}>
                    <img src={neutralIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Neutral</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#FFB200" }}>Neutral</span>{" "}
                </div>
            ),
        },
        {
            value: "negative",
            label: (
                <div className={classes.textOverflow}>
                    <img src={negativeIcon} height="10px" width="10px" alt="" />{" "}
                    <span style={{ padding: "8px" }}>Negative</span>{" "}
                </div>
            ),
            labelWithoutIcon: (
                <div className={classes.textOverflow}>
                    <span style={{ color: "#FF4239" }}>Negative</span>{" "}
                </div>
            ),
        },
    ];
};
export const ButtonsDataRules = () => {
    return [
        {
            label: "Extra words by machine",
            start: "[]",
            end: "[/]",
        },
        {
            label: "Missed word by machine",
            start: "[-]",
            end: "[/-]",
        },
        {
            label: "Speaker Separation",
            start: "[sp]",
            end: "[sp]",
        },
        {
            label: "Machine skipped to transcribe the sentence",
            start: "[$]",
            end: "[/$]",
        },
        {
            label: "Misinterpreted word",
            start: "< >",
            end: "[#]<Enter correct word>[/#]</>",
        },
        {
            label: "Misinterpreted sentence",
            start: "()",
            end: "[#]<Enter correct sentence>[/#](/)",
        },
        {
            label: "Similar way pronounced",
            start: "[@]",
            end: "[#]<Enter correct word>[/#][/@]",
        },
        {
            label: "Word recorded wrong w.r.t spelling",
            start: "[sc]",
            end: "[#]<Enter correct word>[/#][/sc]",
        },
    ];
};

//Summary masterData
export const SummaryOptionsData = () => {
    return [
        { value: "Actions", label: "Action" },
        { value: "Escalations", label: "Escalation" },
        { value: "DeadLines", label: "Deadline" },
        { value: "Appreciations", label: "Appreciation" },
        { value: "Asks", label: "Ask" },
        { value: "Recommendations", label: "Recommendation" },
        { value: "Updates", label: "Update" },
    ];
};
export const SummarySentimentData = () => {
    return [
        { value: "Positive", label: "positive" },
        { value: "Neutral", label: "neutral" },
        { value: "Negative", label: "negative" },
    ];
};
export const SummaryEntitiesOptionsData = () => {
    return [
        { value: "Date", label: "Date" },
        { value: "Event", label: "Event" },
        { value: "Name", label: "Name" },
        { value: "Anthem tool", label: "Anthem Tools" },
        { value: "Tool", label: "Tool" },
        { value: "Technical platform", label: "Technical Platform" },
        { value: "Scrum", label: "Scrum" },
        { value: "Status", label: "Status" },
        { value: "Organization", label: "Organization" },
        { value: "Team name", label: "Team Name" },
        { value: "Technology", label: "Technologies" },
    ];
};

export const SummarycolorCheckEntity = (type) => {
    switch (type) {
        case "Date":
            return "#d42bab";
        case "Event":
            return "#00c7ab";
        case "Name":
        case "Person Name":
            return "#a4460d";
        case "AnthemTools":
        case "Anthem Tools":
        case "Anthemtool":
            return "#f20623";
        case "Tool":
            return "#007000";
        case "Technicalplatform":
        case "Technical platform":
            return "#5e109f";
        case "Scrum":
            return "#ea17af";
        case "Status":
            return "#00559d";
        case "Updates":
            return "#286ce2";
        case "Team Name":
        case "Team name":
            return "#BB8FCE";
        case "Organization":
            return "#2ECC71";
        case "Technologies":
        case "Technology":
            return "#FF5733";
        default:
            return "";
    }
};

export const ColorCheck = (type) => {
    switch (type) {
        case "Actions":
            return "#116fdf";
        case "Escalations":
            return "#839b32";
        case "Deadlines":
            return "#f15057";
        case "Appreciations":
            return "#36a0ff";
        case "Asks":
            return "#00e4d9";
        case "Recommendations":
            return "#35d438";
        case "Call Outs":
            return "#ff00c9";
        case "Date":
            return "#d42bab";
        case "Event":
            return "#00c7ab";
        case "Name":
        case "Person Name":
            return "#a4460d";
        case "AnthemTools":
        case "Anthem Tools":
        case "Anthem tool":
        case "Anthemtool":
            return "#f20623";
        case "Tools":
        case "Tool":
            return "#007000";
        case "TechnicalPlatform":
        case "Technical Platform":
        case "Technical platform":
        case "Technicalplatform":
            return "#5e109f";
        case "Scrum":
            return "#ea17af";
        case "Status":
            return "#00559d";
        case "Positive":
            return "#34B53A";
        case "Neutral":
            return "#FFB200";
        case "Negative":
            return "#FF4239";
        case "Mentoring & Engagement":
            return "#80AD00";
        case "Proactiveness":
            return "#E86933";
        case "Action Plan Tracking":
            return "#0D9098";
        case "Collabaration":
            return "#25BE6B";
        case "Updates":
            return "#286ce2";
        case "Team Name":
        case "Team name":
            return "#BB8FCE";
        case "Organization":
            return "#2ECC71";
        case "Technologies":
        case "Technology":
            return "#FF5733";
        default:
            return "";
    }
};

const MasterData = {
    FilterOptions,
    MakerdataOptions,
    EntitiesDataOptions,
    LabelDataOptions,
    SentimentDataOptions,
    ButtonsDataRules,
    SummaryOptionsData,
    SummarySentimentData,
    SummaryEntitiesOptionsData,
    SummarycolorCheckEntity,
    ColorCheck,
};
export default MasterData;
