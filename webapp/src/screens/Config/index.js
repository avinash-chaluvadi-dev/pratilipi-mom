import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Box } from "@material-ui/core";
import Card from "@material-ui/core/Card";
import Header from "./components/header";
import NoData from "./components/nodata";
import useStyles from "./useStyles";
import ListView from "./components/listview";
import GridView from "./components/gridview";
import Modal from "components/Modal";
import Form, { FormTitle } from "./components/form";
import MoreOptions from "./components/moreOptions";
import Popover from "components/Popover";
import DeleteTeam, {
    DeletTeamActions,
    DeleteTeamTitle,
} from "./components/deleteTeam";
import { getAllTeams } from "store/action/config";
let dataRes = [
    {
        id: 1,

        name: "STARS",

        dl_email: "stars@legato.com",

        sme_name: "A",

        sme_email: "A@legato.com",

        sme_email_notification: false,

        po_name: "B",

        po_email: "B@legato.com",

        po_email_notification: false,

        manager_name: "C",

        manager_email: "C@legato.com",

        manager_email_notification: false,

        created_date: "2022-03-02",

        team_members: [],

        jira_details: {
            ticket_no: "ABCD",
        },
    },
    {
        id: 2,

        name: "HIVE",

        dl_email: "hive@legato.com",

        sme_name: "A",

        sme_email: "A@legato.com",

        sme_email_notification: false,

        po_name: "B",

        po_email: "B@legato.com",

        po_email_notification: false,

        manager_name: "C",

        manager_email: "C@legato.com",

        manager_email_notification: false,

        created_date: "2022-03-02",

        team_members: [],

        jira_details: {
            ticket_no: "ABCD",
        },
    },
    {
        id: 3,

        name: "RPS",

        dl_email: "RPS@legato.com",

        sme_name: "A",

        sme_email: "A@legato.com",

        sme_email_notification: false,

        po_name: "B",

        po_email: "B@legato.com",

        po_email_notification: false,

        manager_name: "C",

        manager_email: "C@legato.com",

        manager_email_notification: false,

        created_date: "2022-03-02",

        team_members: [],

        jira_details: {
            ticket_no: "ABCD",
        },
    },
    {
        id: 4,

        name: "MemberAiops",

        dl_email: "RPS@legato.com",

        sme_name: "A",

        sme_email: "A@legato.com",

        sme_email_notification: false,

        po_name: "B",

        po_email: "B@legato.com",

        po_email_notification: false,

        manager_name: "C",

        manager_email: "C@legato.com",

        manager_email_notification: false,

        created_date: "2022-03-02",

        team_members: [],

        jira_details: {
            ticket_no: "ABCD",
        },
    },
];
const Config = () => {
    const classes = useStyles();
    const [draftView, setDraftView] = useState(false);
    const [isListView, setIsListView] = useState(true);
    const [openModal, setModalOpen] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [isClone, setIsclone] = useState(false);
    const [currentTeam, setCurrentTeam] = useState({});
    const [anchorEl, setAnchorEl] = useState(null);
    const [isDelete, setIsDelete] = useState(false);
    const [data, setData] = useState([]);
    const [filter, setFilter] = useState("");
    const [filterSme, setFilterSme] = useState("");
    const [filterPo, setFilterPo] = useState("");
    const [filterMng, setFilterMng] = useState("");
    const [filterDate, setFilterDate] = useState("");
    const dispatch = useDispatch();
    const { teams, drafts, teamsUptoDate, closeModal } = useSelector(
        (state) => state.configReducer
    );

    useEffect(() => {
        // setData(draftView ? drafts : teams);
        setData(teams);
    }, [draftView, drafts, teams]);

    useEffect(() => {
        dispatch(getAllTeams());
    }, [teamsUptoDate, dispatch]);

    useEffect(() => {
        if (closeModal) setModalOpen(false);
    }, [closeModal]);

    const handleEdit = (x) => {
        setIsDelete(false);
        setIsEditing(true);
        setCurrentTeam(x);
        setModalOpen(true);
    };

    const handleClone = () => {
        setIsDelete(false);
        setIsclone(true);
        setIsEditing(false);
        setModalOpen(true);
        setAnchorEl(null);
    };

    const handleDelete = () => {
        setIsclone(false);
        setIsEditing(false);
        setModalOpen(true);
        setAnchorEl(null);
        setIsDelete(true);
    };

    const handleAdd = () => {
        setIsEditing(false);
        setIsclone(false);
        setIsDelete(false);
        setModalOpen(true);
    };

    const setSearch = (field) => {
        setFilter(field);
    };

    function search(items) {
        return items.filter(
            (item) =>
                item.name
                    .toString()
                    .toLowerCase()
                    .includes(filter.toString().toLowerCase()) +
                    item.manager_name
                        .toString()
                        .toLowerCase()
                        .includes(filter.toString().toLowerCase()) +
                    item.po_name
                        .toString()
                        .toLowerCase()
                        .includes(filter.toString().toLowerCase()) +
                    item.sme_name
                        .toString()
                        .toLowerCase()
                        .includes(filter.toString().toLowerCase()) &&
                (filterSme !== ""
                    ? item.sme_name.toString().toLowerCase() ===
                      filterSme.toString().toLowerCase()
                    : "true") &&
                (filterPo !== ""
                    ? item.po_name.toString().toLowerCase() ===
                      filterPo.toString().toLowerCase()
                    : "true") &&
                (filterMng !== ""
                    ? item.manager_name.toString().toLowerCase() ===
                      filterMng.toString().toLowerCase()
                    : "true") &&
                (filterDate !== ""
                    ? item.created_date.toString().toLowerCase() ===
                      filterDate.toString().toLowerCase()
                    : "true")
        );
    }

    const modalContent = isDelete ? (
        <DeleteTeam handleCancel={setModalOpen} />
    ) : (
        <Form
            currentTeam={currentTeam}
            isEditing={isEditing}
            isClone={isClone}
            handleCancel={setModalOpen}
        />
    );

    const modalActions = isDelete ? (
        <DeletTeamActions
            handleCancel={() => setModalOpen(false)}
            id={currentTeam.id}
        />
    ) : (
        ""
    );

    const modalTitle = isDelete ? (
        <DeleteTeamTitle />
    ) : (
        <FormTitle isEditing={isEditing} />
    );

    return (
        <Box className={classes.MainContent}>
            <Card style={{ padding: "15px" }} className={classes.MainContent}>
                <Header
                    setDraftView={setDraftView}
                    draftView={draftView}
                    setIsListView={setIsListView}
                    handleAdd={handleAdd}
                    setSearch={setSearch}
                    data={data}
                    isListView={isListView}
                    count={search(data).length}
                    filterSme={filterSme}
                    filterPo={filterPo}
                    filterMng={filterMng}
                    filterDate={filterDate}
                    setFilterDate={setFilterDate}
                    setFilterSme={setFilterSme}
                    setFilterMng={setFilterMng}
                    setFilterPo={setFilterPo}
                />
                {data.length === 0 ? (
                    <NoData />
                ) : (
                    <Box marginTop="50px">
                        {isListView ? (
                            <ListView
                                data={search(data)}
                                handleEdit={handleEdit}
                                isClone={isClone}
                                openMoreOption={setAnchorEl}
                                handleClone={handleClone}
                                setCurrentTeam={setCurrentTeam}
                            />
                        ) : (
                            <GridView
                                data={search(data)}
                                handleEdit={handleEdit}
                                isClone={isClone}
                                openMoreOption={setAnchorEl}
                                setAnchorEl={setAnchorEl}
                                setCurrentTeam={setCurrentTeam}
                            />
                        )}
                    </Box>
                )}
            </Card>
            <Modal
                title={modalTitle}
                content={modalContent}
                actions={modalActions}
                open={openModal}
                width={isDelete ? "sm" : "xl"}
                handleClose={() => setModalOpen(false)}
                titleStyle={
                    isDelete
                        ? {
                              height: "30px",
                              background: "#f7f7f7 padding-box",
                          }
                        : {}
                }
            />
            <Popover
                content={
                    <MoreOptions
                        handleClone={handleClone}
                        handleDelete={handleDelete}
                    />
                }
                anchorEl={anchorEl}
                handleClose={() => setAnchorEl(null)}
            />
        </Box>
    );
};

export default Config;
