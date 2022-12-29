import React from "react";
import { Card, Box, Typography, Button } from "@material-ui/core";
import TeamsIcon from "static/Icons/teams.svg";
import useStyles from "../useStyles";
import MoreVertIcon from "@mui/icons-material/MoreVert";

const ListView = ({ data, handleEdit, openMoreOption, setCurrentTeam }) => {
    const classes = useStyles();
    return data.map((team, i) => (
        <Card key={i} className={classes.TeamListCard}>
            <Box display="flex" justifyContent="flex-start">
                <Box className={classes.TeamIconBox}>
                    <img
                        src={TeamsIcon}
                        alt=""
                        style={{
                            height: "22.7px",
                            width: "32px",
                            marginTop: "25px",
                        }}
                    />
                </Box>
                <Box className={classes.TeamCardPartition}>
                    <Box
                        style={{ marginBottom: "5px" }}
                        display="flex"
                        flexDirection="column"
                        alignItems="flex-start"
                    >
                        <Typography className={classes.TeamName}>
                            {team.name}
                        </Typography>
                        <Typography className={classes.Members}>
                            {team?.team_members?.length}
                            {"  "}
                            {team?.team_members?.length === 1
                                ? "member"
                                : "members"}
                        </Typography>
                    </Box>
                    <Typography className={classes.CreatedOn}>
                        Created on: {team.created_date}
                    </Typography>
                </Box>
            </Box>
            <Box display="flex" flexDirection="column" alignItems="flex-start">
                <Typography className={classes.DetailHeader}>
                    Scrum Master/SME:
                </Typography>
                <Typography className={classes.Members}>
                    {team.sme_name}
                </Typography>
            </Box>
            <Box display="flex" flexDirection="column" alignItems="flex-start">
                <Typography className={classes.DetailHeader}>
                    Product Owner:
                </Typography>
                <Typography className={classes.Members}>
                    {team.po_name}
                </Typography>
            </Box>
            <Box display="flex" flexDirection="column" alignItems="flex-start">
                <Typography className={classes.DetailHeader}>
                    Manager:
                </Typography>
                <Typography className={classes.Members}>
                    {team.manager_name}
                </Typography>
            </Box>
            <Button
                variant="outlined"
                color="primary"
                size="medium"
                component="label"
                className={classes.EditButton}
                onClick={() => handleEdit(team)}
            >
                Edit
            </Button>
            <Box
                marginBottom="50px"
                color="#949494"
                style={{ cursor: "pointer" }}
                onClick={(e) => {
                    openMoreOption(e.currentTarget);
                    setCurrentTeam(team);
                }}
            >
                <MoreVertIcon />
            </Box>
        </Card>
    ));
};

export default ListView;
