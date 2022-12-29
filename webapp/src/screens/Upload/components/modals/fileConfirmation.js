import React, { useEffect, useState } from "react";
import Modal from "components/Modal";
import Button from "@material-ui/core/Button";
import Box from "@material-ui/core/Box";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import { withStyles } from "@material-ui/core/styles";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Checkbox from "@mui/material/Checkbox";
import FormHelperText from "@material-ui/core/FormHelperText";
import FormControl from "@material-ui/core/FormControl";
import MenuItem from "@material-ui/core/MenuItem";
import Select from "@material-ui/core/Select";
import useStyles from "../../useStyles";
import { getScrumTeamName } from "store/action/scrumTeam";
import { useSelector, useDispatch } from "react-redux";
import select from "static/images/selectIcon.png";
import clsx from "clsx";

const FileConfirmation = ({
    openFileConfirmation,
    handleFileConfirmationClose,
    file,
    teamName,
    setTeamName,
    submitAction,
}) => {
    const dispatch = useDispatch();
    const [checked, setChecked] = useState(false);
    const [error, setError] = useState(false);
    const { scrumTeams } = useSelector((state) => state.scrumTeamNameReducer);

    useEffect(() => {
        dispatch(getScrumTeamName());
    }, [dispatch]);

    const handleChecked = () => {
        if (teamName !== "") {
            setChecked(!checked);
        } else {
            setError(true);
        }
    };

    function createData(file_name, type, size) {
        return { file_name, type, size };
    }
    const fileName = file.name.substr(0, file.name.lastIndexOf("."));
    const fileSize = file.size;
    const fileExtension = file.name.substr(file.name.lastIndexOf("."));
    function formatSizeUnits(bytes) {
        if (bytes >= 1073741824) {
            bytes = (bytes / 1073741824).toFixed(2) + " GB";
        } else if (bytes >= 1048576) {
            bytes = (bytes / 1048576).toFixed(2) + " MB";
        } else if (bytes >= 1024) {
            bytes = (bytes / 1024).toFixed(2) + " KB";
        } else if (bytes > 1) {
            bytes = bytes + " bytes";
        } else if (bytes === 1) {
            bytes = bytes + " byte";
        } else {
            bytes = "0 bytes";
        }
        return bytes;
    }
    const row = createData(fileName, fileExtension, formatSizeUnits(fileSize));
    const StyledTableCell = withStyles((theme) => ({
        head: {
            color: "#333333",
            fontSize: "14px",
            fontWeight: "bold",
            fontFamily: "Lato",
            paddingTop: "75px",
            paddingLeft: "2px",
            paddingBottom: "10px",
            borderBottom: "none",
            maxWidth: "350px",
            minWidth: "125px",
        },
        body: {
            color: "#333333",
            fontSize: "16px",
            fontFamily: "Lato",
            padding: "2px",
            paddingRight: "20px",
            paddingBottom: "30px",
            borderBottom: "none",
            verticalAlign: "center",
            maxWidth: "350px",
            minWidth: "125px",
        },
    }))(TableCell);

    const classes = useStyles();

    const CustomSelectIcon = withStyles()(({ className, ...rest }) => {
        return (
            <img
                src={select}
                alt="select"
                {...rest}
                className={clsx(className, classes.selectIcon)}
            />
        );
    });

    const title = (
        <Box>
            <Box
                style={{
                    color: "#333333",
                    fontSize: "20px",
                    fontWeight: "bold",
                    lineHeight: 2,
                    fontFamily: "Lato",
                }}
            >
                Upload Recording Confirmation
            </Box>
            <Box
                style={{
                    color: "#333333",
                    fontSize: "16px",
                    fontWeight: "normal",
                    fontFamily: "Lato",
                }}
            >
                Select scrum teamname to proceed further
            </Box>
        </Box>
    );
    const userExists = (obj, value) =>
        obj.team_members.some((user) => user.email === value);
    const content = (
        <div>
            <TableContainer>
                <Table className={classes.table} aria-label="simple table">
                    <TableHead>
                        <TableRow hover classes={{ hover: classes.rowHover }}>
                            <StyledTableCell>Recording Name</StyledTableCell>
                            <StyledTableCell>Scrum Teamname</StyledTableCell>
                            <StyledTableCell>Type</StyledTableCell>
                            <StyledTableCell>Size</StyledTableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        <TableRow
                            key={row.file_name}
                            hover
                            classes={{ hover: classes.rowHover }}
                        >
                            <StyledTableCell component="th" scope="row">
                                {row.file_name}
                            </StyledTableCell>
                            <StyledTableCell component="th" scope="row">
                                <FormControl
                                    className={classes.formControl}
                                    error={error}
                                    variant="outlined"
                                >
                                    <Select
                                        value={teamName}
                                        onChange={(e) => {
                                            setTeamName(e.target.value);
                                            setError(false);
                                        }}
                                        disableUnderline
                                        displayEmpty
                                        className={classes.select}
                                        IconComponent={CustomSelectIcon}
                                        renderValue={
                                            teamName !== ""
                                                ? undefined
                                                : () => "Select team"
                                        }
                                        MenuProps={{
                                            anchorOrigin: {
                                                vertical: "bottom",
                                                horizontal: "left",
                                            },
                                            getContentAnchorEl: null,
                                        }}
                                    >
                                        {scrumTeams?.map((team) => (
                                            <MenuItem
                                                key={team.id}
                                                value={team.id}
                                                style={{
                                                    display: userExists(
                                                        team,
                                                        localStorage.getItem(
                                                            "email"
                                                        )
                                                    )
                                                        ? "flex"
                                                        : "none",
                                                }}
                                                className={classes.menu}
                                            >
                                                {/* <DoneIcon />  */}
                                                {userExists(
                                                    team,
                                                    localStorage.getItem(
                                                        "email"
                                                    )
                                                ) && team.name}
                                            </MenuItem>
                                        ))}
                                    </Select>
                                    {error ? (
                                        <FormHelperText
                                            style={{
                                                color: "#ff0000",
                                                fontSize: "12px",
                                                fontWeight: "normal",
                                            }}
                                        >
                                            *Select a teamname
                                        </FormHelperText>
                                    ) : null}
                                </FormControl>
                            </StyledTableCell>

                            <StyledTableCell>{row.type}</StyledTableCell>
                            <StyledTableCell>{row.size}</StyledTableCell>
                        </TableRow>
                    </TableBody>
                </Table>
            </TableContainer>
            <Box className={classes.checkBox}>
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={checked}
                            onChange={handleChecked}
                            name="checkedB"
                        />
                    }
                    label="I agree that, this recording does not contain any PHI/PII information"
                />
            </Box>
        </div>
    );
    const actions = (
        <Box
            display="flex"
            justifyContent="space-between"
            width="100%"
            color="primary"
        >
            <Box
                style={{
                    color: "#333333",
                    fontSize: "14px",
                    fontWeight: "normal",
                    fontFamily: "Lato",
                }}
            >
                *This process will support audio/video records in formats like
                <Box
                    style={{
                        color: "#333333",
                        fontSize: "16px",
                        fontWeight: "bold",
                        fontFamily: "Lato",
                    }}
                    display="inline"
                >
                    {" "}
                    .mp4/.mp3/.wav
                </Box>{" "}
                only
            </Box>
            <Button
                onClick={submitAction}
                variant="contained"
                disabled={checked ? false : true}
                size="large"
                style={{
                    textTransform: "none",
                    maxWidth: "400px",
                    maxHeight: "40px",
                    minWidth: "175px",
                    minHeight: "40px",
                    fontSize: "16px",
                    fontWeight: "bold",
                    fontFamily: "Lato",
                    color: "#FFFFFF",
                    backgroundColor: checked ? "#1665DF" : "#9bbaeb",
                    borderRadius: "8px",
                }}
            >
                Submit
            </Button>
        </Box>
    );

    return (
        <Box>
            <Modal
                title={title}
                content={content}
                actions={actions}
                width="md"
                open={openFileConfirmation}
                handleClose={handleFileConfirmationClose}
            />
        </Box>
    );
};

export default FileConfirmation;
