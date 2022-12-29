import React, { useState } from "react";
import { Box, Typography, Button, Menu } from "@material-ui/core";
import useStyles from "../useStyles";
import { format } from "date-fns";
import Select, { components } from "react-select";
import select from "static/images/selectIcon.png";
import AutorenewIcon from "@mui/icons-material/Autorenew";
import FilterIcon from "static/Icons/filter.svg";
import { useTheme } from "@mui/material/styles";

import RangeDP from "components/RangeDP";

const Filter = ({
    data,
    filterSme,
    filterPo,
    filterMng,
    filterDate,
    setFilterDate,
    setFilterSme,
    setFilterPo,
    setFilterMng,
}) => {
    const [open, setOpen] = useState(false);
    const [smeValue, setSmeValue] = useState("");
    const [poValue, setPoValue] = useState("");
    const [mngValue, setMngValue] = useState("");
    const [dateValue, setDateValue] = useState("");
    const [unformatedDateValue, setUnformatedDateValue] = useState("");
    const classes = useStyles();
    const theme = useTheme();

    const smeList = [...new Set(data.map((item) => item.sme_name))].map(
        (item, index) => {
            return {
                label: item,
                value: item,
                key: index,
            };
        }
    );

    const mngList = [...new Set(data.map((item) => item.manager_name))].map(
        (item, index) => {
            return {
                label: item,
                value: item,
                key: index,
            };
        }
    );

    const poList = [...new Set(data.map((item) => item.po_name))].map(
        (item, index) => {
            return {
                label: item,
                value: item,
                key: index,
            };
        }
    );

    const DropdownIndicator = (props) => {
        return (
            <components.DropdownIndicator {...props}>
                <img src={select} alt="select" />
            </components.DropdownIndicator>
        );
    };

    const poChange = (e) => {
        setPoValue(e.value);
    };

    const smeChange = (e) => {
        setSmeValue(e.value);
    };

    const mngChange = (e) => {
        setMngValue(e.value);
    };

    const dateChange = (data) => {
        setUnformatedDateValue(new Date(data));
        setDateValue(format(new Date(data), "yyyy-MM-dd"));
    };

    const cancelFilter = () => {
        setOpen(false);
        setSmeValue(filterSme);
        setPoValue(filterPo);
        setMngValue(filterMng);
        setDateValue(filterDate);
    };

    const currentDate = () => {
        return unformatedDateValue === "" ? undefined : unformatedDateValue;
    };

    const clearFilter = () => {
        setOpen(false);
        setFilterMng("");
        setFilterPo("");
        setFilterSme("");
        setFilterDate("");
        setSmeValue("");
        setPoValue("");
        setMngValue("");
        setDateValue("");
        setUnformatedDateValue("");
    };

    const applyFilter = () => {
        setOpen(false);
        setFilterMng(mngValue);
        setFilterPo(poValue);
        setFilterSme(smeValue);
        setFilterDate(dateValue);
    };

    return (
        <>
            <Button
                variant="outlined"
                className={classes.filterbtn}
                onClick={(e) => setOpen(e.currentTarget)}
                style={{ textTransform: "none" }}
            >
                <Box
                    component="img"
                    src={FilterIcon}
                    height="20px"
                    width="20px"
                    alt=""
                    style={{ marginRight: "5px" }}
                />
                Filter
            </Button>
            <Box>
                <Menu
                    id="basic-menu"
                    anchorEl={open}
                    open={Boolean(open)}
                    onClose={cancelFilter}
                    className={classes.menuWidth}
                >
                    <Box
                        component="form"
                        sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            flexDirection: "column",
                        }}
                    >
                        <Box display="flex" justifyContent="space-between">
                            <Box display="flex" flexDirection="column">
                                <Typography
                                    variant="p"
                                    component="p"
                                    className={classes.LabelMargin}
                                >
                                    Scrum Master/SME
                                </Typography>
                                <Select
                                    className={classes.select}
                                    components={{
                                        DropdownIndicator,
                                        IndicatorSeparator: () => null,
                                    }}
                                    value={smeList.filter(function (option) {
                                        return option.value === smeValue;
                                    })}
                                    onChange={smeChange}
                                    options={smeList}
                                    isSearchable={false}
                                    label="Single select"
                                ></Select>
                            </Box>
                            <Box display="flex" flexDirection="column">
                                <Typography
                                    variant="p"
                                    component="p"
                                    className={classes.LabelMargin}
                                >
                                    Product Owner
                                </Typography>
                                <Select
                                    className={classes.select}
                                    components={{
                                        DropdownIndicator,
                                        IndicatorSeparator: () => null,
                                    }}
                                    value={poList.filter(function (option) {
                                        return option.value === poValue;
                                    })}
                                    onChange={poChange}
                                    options={poList}
                                    isSearchable={false}
                                ></Select>
                            </Box>
                        </Box>
                        <Box
                            display="flex"
                            justifyContent="space-between"
                            mt={2}
                        >
                            <Box display="flex" flexDirection="column">
                                <Typography
                                    variant="p"
                                    component="p"
                                    className={classes.LabelMargin}
                                >
                                    Manager
                                </Typography>
                                <Select
                                    className={classes.select}
                                    components={{
                                        DropdownIndicator,
                                        IndicatorSeparator: () => null,
                                    }}
                                    value={mngList.filter(function (option) {
                                        return option.value === mngValue;
                                    })}
                                    onChange={mngChange}
                                    options={mngList}
                                    isSearchable={false}
                                ></Select>
                            </Box>
                            <Box
                                display="flex"
                                flexDirection="column"
                                style={{ marginRight: "6px" }}
                            >
                                <Typography
                                    variant="p"
                                    component="p"
                                    className={classes.LabelMargin}
                                >
                                    Date
                                </Typography>
                                <RangeDP
                                    inputWidth={36}
                                    currentValue={currentDate()}
                                    handleDateChange={(data) =>
                                        dateChange(data)
                                    }
                                />
                            </Box>
                        </Box>
                    </Box>

                    <Box
                        component="div"
                        sx={{
                            display: "flex",
                            flexWrap: "wrap",
                            justifyContent: "space-between",
                            marginTop: "30px",
                            borderTop: "1px solid #ccc",
                            paddingTop: "10px",
                        }}
                    >
                        <Button
                            color="secondary"
                            onClick={clearFilter}
                            startIcon={<AutorenewIcon />}
                            className={classes.ResetButton}
                        >
                            Reset All Filters
                        </Button>
                        <Box
                            display="flex"
                            justifyContent="space-between"
                            width="60%"
                        >
                            <Button
                                autoFocus
                                onClick={cancelFilter}
                                variant="outlined"
                                color="primary"
                                className={classes.SubmitButton}
                                style={{
                                    color: theme.palette.primary.main,
                                    backgroundColor: "#FFFFFF",
                                }}
                            >
                                Cancel
                            </Button>
                            <Button
                                autoFocus
                                color="primary"
                                onClick={applyFilter}
                                variant="contained"
                                className={classes.SubmitButton}
                            >
                                Apply Filters
                            </Button>
                        </Box>
                    </Box>
                </Menu>
            </Box>
        </>
    );
};

export default Filter;
