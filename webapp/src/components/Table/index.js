import React, { useState } from "react";
import MaterialTable, { MTableToolbar, MTablePagination } from "material-table";
import { forwardRef } from "react";
import AddBox from "@material-ui/icons/AddBox";
import Check from "@material-ui/icons/Check";
import ChevronLeft from "@material-ui/icons/ChevronLeft";
import ChevronRight from "@material-ui/icons/ChevronRight";
import Clear from "@material-ui/icons/Clear";
import DeleteOutline from "@material-ui/icons/DeleteOutline";
import Edit from "@material-ui/icons/Edit";
import FilterList from "@material-ui/icons/FilterList";
import FirstPage from "@material-ui/icons/FirstPage";
import LastPage from "@material-ui/icons/LastPage";
import Remove from "@material-ui/icons/Remove";
import SaveAlt from "@material-ui/icons/SaveAlt";
import Search from "@material-ui/icons/Search";
import ViewColumn from "@material-ui/icons/ViewColumn";
import SaveIcon from "@material-ui/icons/Save";
import { useTheme } from "@material-ui/core/styles";
import sort from "static/images/sort.png";

const tableIcons = {
    Save: forwardRef((props, ref) => <SaveIcon {...props} ref={ref} />),
    Add: forwardRef((props, ref) => <AddBox {...props} ref={ref} />),
    Check: forwardRef((props, ref) => <Check {...props} ref={ref} />),
    Clear: forwardRef((props, ref) => <Clear {...props} ref={ref} />),
    Delete: forwardRef((props, ref) => <DeleteOutline {...props} ref={ref} />),
    DetailPanel: forwardRef((props, ref) => (
        <ChevronRight {...props} ref={ref} />
    )),
    Edit: forwardRef((props, ref) => <Edit {...props} ref={ref} />),
    Export: forwardRef((props, ref) => <SaveAlt {...props} ref={ref} />),
    Filter: forwardRef((props, ref) => <FilterList {...props} ref={ref} />),
    FirstPage: forwardRef((props, ref) => <FirstPage {...props} ref={ref} />),
    LastPage: forwardRef((props, ref) => <LastPage {...props} ref={ref} />),
    NextPage: forwardRef((props, ref) => <ChevronRight {...props} ref={ref} />),
    PreviousPage: forwardRef((props, ref) => (
        <ChevronLeft {...props} ref={ref} />
    )),
    ResetSearch: forwardRef((props, ref) => <Clear {...props} ref={ref} />),
    Search: forwardRef((props, ref) => <Search {...props} ref={ref} />),
    SortArrow: forwardRef((props, ref) => (
        <img src={sort} alt="sort" {...props} ref={ref} />
    )),
    ThirdStateCheck: forwardRef((props, ref) => (
        <Remove {...props} ref={ref} />
    )),
    ViewColumn: forwardRef((props, ref) => <ViewColumn {...props} ref={ref} />),
};

const TableComponent = ({
    data,
    columns,
    title,
    filtering,
    exportButton,
    toolbar,
    hideToolbar,
    searchLeftAlign,
    dense,
    headerStyle,
    disablePaging,
    longTable,
    ...props
}) => {
    const tableRef = React.createRef();
    const theme = useTheme();
    const [selectedRow, setSelectedRow] = useState(false);
    const customFun = (evt, selectedRowVal) => {
        setSelectedRow(selectedRowVal.tableData.id);
        props.from !== "actionTab" &&
            props.redirectFun(
                selectedRowVal.tableData.id,
                data[selectedRowVal.tableData.id]
            );
    };
    const defaultHeaderStyle = {
        color: "#999999",
        fontSize: 14,
        fontWeight: 400,
    };
    return (
        <MaterialTable
            tableRef={tableRef}
            icons={tableIcons}
            title={title}
            columns={columns}
            data={data}
            // isLoading={true}
            filtering={filtering}
            components={{
                Toolbar: (props) => (
                    <div>
                        {toolbar}
                        <MTableToolbar {...props} />
                    </div>
                ),
                Pagination: (props) => (
                    <div style={{ overflow: "hidden" }}>
                        <MTablePagination {...props} />
                    </div>
                ),
            }}
            onRowClick={(evt, selectedRow) => customFun(evt, selectedRow)}
            options={{
                searchFieldAlignment: searchLeftAlign ? "left" : "right",
                searchFieldVariant: "outlined",
                searchFieldStyle: {
                    background: theme.palette.white.tertiary,
                    height: "30px",
                },
                align: "left",
                filtering: false,
                toolbar: hideToolbar ? false : true,
                showTitle: true,
                grouping: false,
                exportButton: exportButton,
                titleStyle: { fontSize: 20, fontWeight: "bold" },
                headerStyle: headerStyle ? headerStyle : defaultHeaderStyle,
                padding: dense ? "dense" : "default",
                rowStyle: (x, i) => ({
                    backgroundColor: i % 2 === 0 ? "#f2f2f2" : "#ffffff",
                    pointerEvents:
                        x.status.toLowerCase() === "completed"
                            ? "auto"
                            : x.status.toLowerCase() === "ready for review"
                            ? "auto"
                            : x.status.toLowerCase() ===
                              "user review in progress"
                            ? "auto"
                            : "none",
                }),
                emptyRowsWhenPaging: false,
                paging: disablePaging ? false : true,
                pageSize: longTable ? 10 : 5,
                doubleHorizontalScroll: false,
                sorting: true,
                sortIcon: "show",
            }}
        />
    );
};

export default TableComponent;
