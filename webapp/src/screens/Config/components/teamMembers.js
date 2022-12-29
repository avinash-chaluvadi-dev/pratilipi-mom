import React from "react";
import Input from "components/FormInput/input";
import { Box, Button, Typography } from "@material-ui/core";
// import AddIcon from '@mui/icons-material/Add';
// import ClearOutlinedIcon from '@mui/icons-material/ClearOutlined';
import useGlobalStyles from "styles";
import RemoveIcon from "static/Icons/remove-red.svg";
import AddIcon from "static/Icons/add-blue.svg";
import { useTheme } from "@material-ui/core/styles";

const TeamMembers = ({ inputList, setInputList }) => {
    const theme = useTheme();
    const globalStyles = useGlobalStyles({ ml: "7px" });

    const handleRemoveClick = (index) => {
        const list = [...inputList];
        list.splice(index, 1);
        setInputList(list);
    };
    const handleAddClick = () => {
        setInputList([...inputList, { name: "", email: "" }]);
    };
    const handleInputChange = (e, index) => {
        const { name, value } = e.target;
        const list = [...inputList];
        list[index][name] = value;
        setInputList(list);
    };
    return (
        <Box>
            <Box
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                mt={3}
                mb={2}
                className={globalStyles.TransparentBorder}
            >
                <Box className={globalStyles.SecondHeading} display="flex">
                    <>Team Members</>
                    <Typography
                        variant="body1"
                        className={`${globalStyles.textMuted} ${globalStyles.ml}`}
                    >
                        Optional
                    </Typography>
                </Box>
                <Button
                    variant="outlined"
                    size="medium"
                    color="primary"
                    component="label"
                    onClick={() => handleAddClick()}
                    className={globalStyles.NoBorderButton}
                >
                    <Box
                        component="img"
                        src={AddIcon}
                        style={{ marginRight: "10px" }}
                    />
                    Add Member
                </Button>
            </Box>

            {inputList.map((x, i) => (
                <Box display="flex" alignItems="center" mb={3}>
                    <Input
                        placeholder="Tim Anderson"
                        name="name"
                        value={x.name}
                        onChange={(e) => handleInputChange(e, i)}
                    />
                    <Input
                        ml={3}
                        name="email"
                        placeholder="anderson@abc.com"
                        value={x.email}
                        onChange={(e) => handleInputChange(e, i)}
                        type="email"
                    />
                    <Button
                        style={{ color: theme.palette.error.main }}
                        onClick={() => handleRemoveClick(i)}
                        className={globalStyles.NoBorderButton}
                    >
                        <Box
                            component="img"
                            src={RemoveIcon}
                            style={{ marginRight: "10px" }}
                        />
                        Remove
                    </Button>
                </Box>
            ))}
        </Box>
    );
};

export default TeamMembers;
