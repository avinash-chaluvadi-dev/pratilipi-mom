import React from "react";
import { Card, Box, Typography, Button } from "@material-ui/core";
import TeamsIcon from "static/Icons/teams.svg";
import useStyles from "../useStyles";
import MoreVertIcon from "@mui/icons-material/MoreVert";

const GridView = ({ data, handleEdit, openMoreOption, setCurrentTeam }) => {
    const classes = useStyles();
    return (
        <Box className={classes.GridCardContainer}>
            {data.map((team, i) => (
                <Card key={i} className={classes.TeamGridCard}>
                    <Box
                        display="flex"
                        justifyContent="center"
                        flexDirection="column"
                        alignItems="center"
                    >
                        <Box display="flex" width="250px">
                            <Box
                                className={classes.TeamIconBox}
                                margin="0 0 5px 85px"
                            >
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
                            <Box
                                color="#949494"
                                marginLeft="auto"
                                style={{
                                    cursor: "pointer",
                                }}
                            >
                                <MoreVertIcon
                                    onClick={(e) => {
                                        openMoreOption(e.currentTarget);
                                        setCurrentTeam(team);
                                    }}
                                />
                            </Box>
                        </Box>
                        <Box className={classes.TeamGridCardPartition}>
                            <Box
                                style={{ marginBottom: "5px" }}
                                display="flex"
                                flexDirection="column"
                                alignItems="center"
                                justifyContent="center"
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
                    <Box
                        display="flex"
                        flexDirection="column"
                        alignItems="center"
                        mb={1}
                    >
                        <Typography className={classes.DetailHeader}>
                            Scrum Master/SME:
                        </Typography>
                        <Typography className={classes.Members}>
                            {team.sme_name}
                        </Typography>
                    </Box>
                    <Box
                        display="flex"
                        flexDirection="column"
                        alignItems="center"
                        mb={1}
                    >
                        <Typography className={classes.DetailHeader}>
                            Product Owner:
                        </Typography>
                        <Typography className={classes.Members}>
                            {team.po_name}
                        </Typography>
                    </Box>
                    <Box
                        display="flex"
                        flexDirection="column"
                        alignItems="center"
                        mb={1}
                    >
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
                </Card>
            ))}
            ;
        </Box>
    );
};

export default GridView;
