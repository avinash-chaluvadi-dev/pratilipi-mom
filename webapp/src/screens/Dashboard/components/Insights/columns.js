import PerformanceMetric from "./performanceMetric";
import RecommendedTalktime from "./recommendedTalktime";
import { Typography } from "@material-ui/core";

const columns = [
    {
        title: "Active Users",
        field: "name",
        width: "15%",
        cellStyle: {
            borderBottom: "none",
            fontWeight: "600",
        },
        render: (props) => (
            <>
                <Typography style={{ fontWeight: 600 }}>
                    {props.name}
                </Typography>
                <Typography
                    style={{ color: "rgb(153, 153, 153)", marginTop: "5px" }}
                >{`${props.no_of_meetings} meetings`}</Typography>
            </>
        ),
    },
    {
        title: "Recommended Talktime",
        field: "talktime",
        width: "25%",
        align: "center",
        cellStyle: {
            borderBottom: "none",
        },
        render: (props) => <RecommendedTalktime data={props.talktime} />,
    },
    {
        title: "Mentoring and Engagement",
        field: "mentoring_and_engagement",
        width: "15%",
        cellStyle: {
            borderBottom: "none",
            paddingLeft: "50px",
        },
        render: (props) => (
            <PerformanceMetric performance={props.mentoring_and_engagement} />
        ),
    },
    {
        title: "Action Plan Tracking",
        field: "action_plan_tracing",
        width: "15%",
        cellStyle: {
            borderBottom: "none",
            paddingLeft: "50px",
        },
        render: (props) => (
            <PerformanceMetric performance={props.action_plan_tracing} />
        ),
    },
    {
        title: "Proactiveness",
        field: "proactiveness",
        width: "15%",
        cellStyle: {
            borderBottom: "none",
        },
        render: (props) => (
            <PerformanceMetric performance={props.proactiveness} />
        ),
    },
    {
        title: "Collabration",
        field: "collaboration",
        width: "15%",
        cellStyle: {
            borderBottom: "none",
        },
        render: (props) => (
            <PerformanceMetric performance={props.collaboration} />
        ),
    },
];

export default columns;

export const data = [
    {
        name: "Mattie Blooman",
        mentoring_and_engagement: "Excellent",
        no_of_meetings: 24,
        action_plan_tracing: "Good",
        proactiveness: "Good",
        collaboration: "Excellent",
        talktime: [
            {
                talktime: 0,
            },
            {
                talktime: 100,
            },
            {
                talktime: 250,
            },
            {
                talktime: 160,
            },
            {
                talktime: 102,
            },
            {
                talktime: 60,
            },
            {
                talktime: 150,
            },
            {
                talktime: 250,
            },
            {
                talktime: 180,
            },
            {
                talktime: 125,
            },
            {
                talktime: 102,
            },
            {
                talktime: 60,
            },
        ],
    },
    {
        name: "Olivia Arribas",
        mentoring_and_engagement: "Good",
        action_plan_tracing: "Excellent",
        no_of_meetings: 20,
        proactiveness: "Poor",
        collaboration: "Excellent",
        talktime: [
            {
                talktime: 200,
            },
            {
                talktime: 20,
            },
            {
                talktime: 0,
            },
            {
                talktime: 160,
            },
            {
                talktime: 300,
            },
            {
                talktime: 0,
            },
            {
                talktime: 225,
            },
            {
                talktime: 150,
            },
        ],
    },
    {
        name: "Graham Griffiths",
        no_of_meetings: 16,
        mentoring_and_engagement: "Poor",
        action_plan_tracing: "Medium",
        proactiveness: "Poor",
        collaboration: "Good",
        talktime: [
            {
                talktime: 140,
            },
            {
                talktime: 220,
            },
            {
                talktime: 60,
            },
            {
                talktime: 180,
            },
            {
                talktime: 120,
            },
            {
                talktime: 0,
            },
            {
                talktime: 221,
            },
            {
                talktime: 90,
            },
        ],
    },
    {
        name: "Natalia Khanwald",
        mentoring_and_engagement: "Medium",
        no_of_meetings: 12,
        action_plan_tracing: "Poor",
        proactiveness: "Excellent",
        collaboration: "Good",
        talktime: [
            {
                talktime: 140,
            },
            {
                talktime: 220,
            },
            {
                talktime: 160,
            },
            {
                talktime: 180,
            },
            {
                talktime: 120,
            },
            {
                talktime: 0,
            },
            {
                talktime: 20,
            },
            {
                talktime: 190,
            },
        ],
    },
    {
        name: "Jonas",
        mentoring_and_engagement: "Good",
        no_of_meetings: 10,
        action_plan_tracing: "Excellent",
        proactiveness: "Medium",
        collaboration: "Poor",
        talktime: [
            {
                talktime: 140,
            },
            {
                talktime: 220,
            },
            {
                talktime: 60,
            },
            {
                talktime: 180,
            },
            {
                talktime: 120,
            },
            {
                talktime: 0,
            },
            {
                talktime: 221,
            },
            {
                talktime: 90,
            },
        ],
    },
    {
        name: "Robert philips",
        mentoring_and_engagement: "Excellent",
        no_of_meetings: 10,
        action_plan_tracing: "Good",
        proactiveness: "Poor",
        collaboration: "Medium",
        talktime: [
            {
                talktime: 200,
            },
            {
                talktime: 40,
            },
            {
                talktime: 0,
            },
            {
                talktime: 100,
            },
            {
                talktime: 220,
            },
            {
                talktime: 80,
            },
            {
                talktime: 225,
            },
            {
                talktime: 150,
            },
        ],
    },
    {
        name: "Andrew Kim",
        mentoring_and_engagement: "Good",
        no_of_meetings: 9,
        action_plan_tracing: "Medium",
        proactiveness: "Poor",
        collaboration: "Medium",
        talktime: [
            {
                talktime: 140,
            },
            {
                talktime: 220,
            },
            {
                talktime: 60,
            },
            {
                talktime: 180,
            },
            {
                talktime: 120,
            },
            {
                talktime: 0,
            },
            {
                talktime: 221,
            },
            {
                talktime: 90,
            },
        ],
    },
    {
        name: "Peter Anderson",
        mentoring_and_engagement: "Medium",
        no_of_meetings: 8,
        action_plan_tracing: "Excellent",
        proactiveness: "Medium",
        collaboration: "Poor",
        talktime: [
            {
                talktime: 200,
            },
            {
                talktime: 210,
            },
            {
                talktime: 190,
            },
            {
                talktime: 240,
            },
        ],
    },
];
