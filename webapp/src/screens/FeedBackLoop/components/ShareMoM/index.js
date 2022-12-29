import React from "react";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import pdfIcon from "static/images/downloadIcon.svg";
import View from "static/images/visibility_black.svg";
import Teams from "static/images/teams.svg";
import Outlook from "static/images/outlook.svg";
import Jira from "static/images/jira.svg";
import useStyles from "./useStyles";

function TabPanel(props) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
        </div>
    );
}

const ShareMoM = ({ downloadPDF, previewPDF }) => {
    const classes = useStyles();
    const [value, setValue] = React.useState(0);
    const handleChange = (event, newValue) => {
        setValue(newValue);
    };
    return (
        <>
            <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
                <Tabs value={value} onChange={handleChange}>
                    <Tab label="Share" style={{ textTransform: "none" }} />
                    <Tab
                        label="Download PDF"
                        style={{ textTransform: "none" }}
                    />
                </Tabs>
            </Box>
            <TabPanel value={value} index={0} style={{ height: "200px" }}>
                <Box
                    display="flex"
                    justifyContent="center"
                    flexDirection="column"
                    alignItems="center"
                >
                    <Typography style={{ marginBottom: "20px" }}>
                        Share the generated MoM
                    </Typography>
                    <Box
                        display="flex"
                        justifyContent="space-around"
                        alignItems="center"
                        width="100%"
                    >
                        <Button
                            variant="outlined"
                            className={classes.shareMoMButton}
                        >
                            <img src={Outlook} alt="outlook icon" />
                            Share to Outlook
                        </Button>
                        <Button
                            variant="outlined"
                            className={classes.shareMoMButton}
                        >
                            <img src={Teams} alt="teams icon" />
                            Share to Teams
                        </Button>
                        <Button
                            variant="outlined"
                            className={classes.shareMoMButton}
                        >
                            <img src={Jira} alt="jira icon" />
                            Share to Jira
                        </Button>
                    </Box>
                </Box>
            </TabPanel>
            <TabPanel value={value} index={1} style={{ height: "200px" }}>
                <Box
                    display="flex"
                    justifyContent="center"
                    flexDirection="column"
                    alignItems="center"
                >
                    <Typography style={{ marginBottom: "20px" }}>
                        Download the MoM in PDF format
                    </Typography>
                    <Button
                        onClick={downloadPDF}
                        variant="outlined"
                        color="primary"
                        style={{ width: "30%", textTransform: "none" }}
                    >
                        <img
                            src={pdfIcon}
                            alt="pdf icon"
                            style={{ marginRight: "10px" }}
                        />
                        Download PDF
                    </Button>
                </Box>
            </TabPanel>
        </>
    );
};

export default ShareMoM;
