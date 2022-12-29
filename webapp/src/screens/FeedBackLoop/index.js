import React from "react";
import { Card, Grid } from "@material-ui/core";
import Feedbackloop from "./components";
import useStyles from "screens/FeedBackLoop/styles";

const FeedBackLoop = () => {
    const classes = useStyles();
    return (
        <Grid xs={12} item={true}>
            <Card className={classes.cardPadding}>
                <Feedbackloop />
            </Card>
        </Grid>
    );
};
export default FeedBackLoop;
