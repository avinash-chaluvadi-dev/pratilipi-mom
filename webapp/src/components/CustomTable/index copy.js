import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import Paper from "@material-ui/core/Paper";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TablePagination from "@material-ui/core/TablePagination";
import TableRow from "@material-ui/core/TableRow";
import Checkbox from "@material-ui/core/Checkbox";
import customTableStyles from "components/CustomTable/customTableStyles";

export default function StickyHeadTable(props) {
    const classes = customTableStyles();
    const { tableData, columns } = props;
    const [page, setPage] = React.useState(0);
    const [rowsPerPage, setRowsPerPage] = React.useState(10);

    const createData = (id, summary, date, assign_to) => {
        return { id, summary, date, assign_to };
    };

    const rows = tableData.map((item) =>
        createData(item.id, item.summary, item.date, item.assign_to)
    );

    const handleChangePage = (event, newPage) => {
        setPage(newPage);
    };

    const handleChangeRowsPerPage = (event) => {
        setRowsPerPage(+event.target.value);
        setPage(0);
    };

    const handleSelectAllClick = (event) => {
        if (event.target.checked) {
            this.setState((state) => ({ selected: true }));
            return;
        }
        this.setState({ selected: [] });
    };

    const handleCheckboxClick = (event, id) => {
        event.stopPropagation();
        console.log("checkbox select");
        // const { selected } = this.state;
        // const selectedIndex = selected.indexOf(id);
        // let newSelected = [];

        // if (selectedIndex === -1) {
        //   newSelected = newSelected.concat(selected, id);
        // } else if (selectedIndex === 0) {
        //   newSelected = newSelected.concat(selected.slice(1));
        // } else if (selectedIndex === selected.length - 1) {
        //   newSelected = newSelected.concat(selected.slice(0, -1));
        // } else if (selectedIndex > 0) {
        //   newSelected = newSelected.concat(
        //     selected.slice(0, selectedIndex),
        //     selected.slice(selectedIndex + 1)
        //   );
        // }
        // this.setState({ selected: newSelected });
    };

    return (
        <Paper className={classes.root} elevation={0}>
            <TableContainer className={classes.container}>
                <Table stickyHeader aria-label="sticky table">
                    <TableHead>
                        <TableRow>
                            {columns.map((column) => (
                                <TableCell
                                    key={column.id}
                                    align={column.align}
                                    style={{ minWidth: column.minWidth }}
                                    className={classes.headerStyle}
                                >
                                    {column.label}
                                </TableCell>
                            ))}
                            <TableCell
                                padding="checkbox"
                                style={{ minWidth: 40, align: "left" }}
                                className={classes.headerStyle}
                            >
                                <Checkbox
                                    // indeterminate={numSelectednumSelected > 0 && numSelected < rowCount}
                                    checked={true}
                                    onChange={handleSelectAllClick}
                                />
                            </TableCell>
                            <TableCell
                                padding="checkbox"
                                style={{ minWidth: 40, align: "left" }}
                                className={classes.headerStyle}
                            >
                                <Checkbox
                                    // indeterminate={numSelectednumSelected > 0 && numSelected < rowCount}
                                    checked={true}
                                    onChange={handleSelectAllClick}
                                />
                            </TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {rows.map((row, index) => {
                            return (
                                <>
                                    <TableRow
                                        hover
                                        role="checkbox"
                                        tabIndex={-1}
                                        key={row.summary}
                                        style={
                                            index % 2
                                                ? { background: "#f7f7f7" }
                                                : { background: "#ffffff" }
                                        }
                                    >
                                        {columns.map((column) => {
                                            const value = row[column.id];
                                            return (
                                                <TableCell
                                                    key={column.id}
                                                    align={column.align}
                                                    className={
                                                        classes.cellStyle
                                                    }
                                                >
                                                    {column.format &&
                                                    typeof value === "number"
                                                        ? column.format(value)
                                                        : value}
                                                </TableCell>
                                            );
                                        })}
                                        <TableCell
                                            className="selectCheckbox"
                                            padding="checkbox"
                                            className={classes.cellStyle}
                                        >
                                            <Checkbox
                                                onClick={(event) =>
                                                    handleCheckboxClick(event)
                                                }
                                                className="selectCheckbox"
                                                checked={true}
                                            />
                                        </TableCell>
                                        <TableCell
                                            className="selectCheckbox"
                                            padding="checkbox"
                                            className={classes.cellStyle}
                                        >
                                            <Checkbox
                                                onClick={(event) =>
                                                    handleCheckboxClick(event)
                                                }
                                                className="selectCheckbox"
                                                checked={true}
                                            />
                                        </TableCell>
                                    </TableRow>
                                </>
                            );
                        })}
                    </TableBody>
                </Table>
            </TableContainer>
            {/* <TablePagination
        rowsPerPageOptions={[5, 10, 15, 20, 25, 50, 100]}
        component="div"
        count={rows.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onChangePage={handleChangePage}
        onChangeRowsPerPage={handleChangeRowsPerPage}
      /> */}
        </Paper>
    );
}
