import React from "react";
import { Box } from "@material-ui/core";
import Typography from "@mui/material/Typography";
import useStyles from "../useStyles";
import Button from "@material-ui/core/Button";
import Paper from "@mui/material/Paper";
import InputBase from "@mui/material/InputBase";
import Filter from "./filter";
import GridIcon from "static/Icons/true-false.png";
import ActiveListIcon from "static/Icons/bulleted-list.png";
import ActiveGridIcon from "static/Icons/grid-active.svg";
import ListIcon from "static/Icons/list-gray.svg";
import SearchIcon from "static/Icons/search.svg";
import AddIcon from "static/Icons/add.svg";
import useGlobalStyles from "styles";

const Header = ({
    setIsListView,
    handleAdd,
    isListView,
    count,
    data,
    setSearch,
    filterSme,
    filterPo,
    filterMng,
    filterDate,
    setFilterDate,
    setFilterSme,
    setFilterPo,
    setFilterMng,
}) => {
    const classes = useStyles();
    const globalStyles = useGlobalStyles();

    return (
        <>
            <Box display="flex" justifyContent="space-between">
                <Typography
                    variant="body"
                    gutterBottom
                    component="div"
                    className={classes.TeamsConfiguration}
                >
                    Team(s) Configuration
                </Typography>
                <Button
                    variant="contained"
                    size="medium"
                    color="primary"
                    component="label"
                    onClick={() => handleAdd()}
                    className={globalStyles.MainButton}
                >
                    <Box
                        component="img"
                        src={AddIcon}
                        alt=""
                        style={{ marginRight: "10px" }}
                    />
                    Add Team
                </Button>
            </Box>
            <Box display="flex" justifyContent="space-between" marginTop="20px">
                <Box display="flex" justifyContent="center">
                    <Box className={classes.TeamCount}>{count} Teams</Box>
                    {/* <Button
            variant={draftView ? 'contained' : 'outlined'}
            color={draftView ? 'primary' : ''}
            size="medium"
            component="label"
            onClick={() => setDraftView(true)}
            className={classes.headerButtons}
            style={{ color: draftView ? 'white' : theme.palette.primary.main }}
          >
            Drafts
          </Button>
          <Button
            variant={draftView ? 'outlined' : 'contained'}
            color={draftView ? '' : 'primary'}
            onClick={() => setDraftView(false)}
            size="medium"
            component="label"
            className={classes.headerButtons}
            style={{ color: draftView ? theme.palette.primary.main : 'white' }}
          >
            All
          </Button> */}
                </Box>
                <Box display="flex" justifyContent="flex-end">
                    <Paper className={classes.SearchInputRoot}>
                        <Box
                            component="img"
                            src={SearchIcon}
                            alt=""
                            style={{ height: "20px", margin: "8px" }}
                        />
                        <InputBase
                            style={{ width: "100%", marginLeft: "5px" }}
                            placeholder="Search"
                            inputProps={{ "aria-label": "Search" }}
                            onChange={(e) => setSearch(e.target.value)}
                        />
                    </Paper>
                    <Filter
                        data={data}
                        filterSme={filterSme}
                        filterPo={filterPo}
                        filterMng={filterMng}
                        filterDate={filterDate}
                        setFilterDate={setFilterDate}
                        setFilterSme={setFilterSme}
                        setFilterMng={setFilterMng}
                        setFilterPo={setFilterPo}
                    />
                    <Box mt={1}>
                        <Box
                            component="img"
                            src={isListView ? GridIcon : ActiveGridIcon}
                            alt=""
                            style={{ height: "20px", cursor: "pointer" }}
                            onClick={() => setIsListView(false)}
                        />
                        <Box
                            component="img"
                            src={isListView ? ActiveListIcon : ListIcon}
                            alt=""
                            style={{
                                height: "20px",
                                marginLeft: "15px",
                                cursor: "pointer",
                            }}
                            onClick={() => setIsListView(true)}
                        />
                    </Box>
                </Box>
            </Box>
        </>
    );
};

export default Header;
