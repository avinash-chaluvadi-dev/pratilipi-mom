import { Box, Grid } from "@material-ui/core";
import useStyles from "screens/FeedBackLoop/styles";

const NoDataFound = (props) => {
    const classes = useStyles();
    return (
        <Grid item xs={props.size} className={classes.actionCardCss}>
            <Box
                style={{
                    fontSize: "20px",
                    margin: props.margin,
                    fontWeight: "bold",
                }}
            >
                No Data Found
            </Box>
        </Grid>
    );
};
export default NoDataFound;
