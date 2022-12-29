import upload from "static/images/uploadn.png";
import dashboard from "static/images/dashboard.png";
import cog from "static/Icons/cog.svg";

const menu = [
    {
        Dashboard: <img src={dashboard} alt="Upload" />,
    },
    {
        Upload: <img src={upload} alt="Upload" />,
    },
    {
        Configuration: <img src={cog} alt="Upload" />,
    },
];

export default menu;
