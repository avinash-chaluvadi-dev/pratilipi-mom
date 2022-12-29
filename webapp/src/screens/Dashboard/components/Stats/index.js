import React from "react";
import Box from "@material-ui/core/Box";
import StatsCard from "./statscard";
import upload from "static/images/uploadn.png";
import attention from "static/images/attention.svg";
import messaging from "static/images/messaging.svg";
import ok from "static/images/ok.svg";
import styles from "screens/Dashboard/styles";
import { useSelector } from "react-redux";

const Stats = () => {
    const classes = styles();
    const { summary } = useSelector((state) => state.summaryReducer);
    var zero = 0;
    return (
        <Box
            display="flex"
            justifyContent="space-between"
            width="100%"
            className={classes.statsCardMargin}
        >
            <StatsCard
                icon={upload}
                count={
                    summary?.dashboard_info?.recordings_uploaded
                        ? summary.dashboard_info.recordings_uploaded.toLocaleString(
                              "en-US",
                              {
                                  minimumIntegerDigits: 2,
                                  useGrouping: false,
                              }
                          )
                        : zero.toLocaleString("en-US", {
                              minimumIntegerDigits: 2,
                              useGrouping: false,
                          })
                }
                footer="Recordings Uploaded"
                color="#286ce2"
            />
            <StatsCard
                icon={attention}
                count={
                    summary?.dashboard_info?.mom_ready_for_review
                        ? summary.dashboard_info.mom_ready_for_review.toLocaleString(
                              "en-US",
                              {
                                  minimumIntegerDigits: 2,
                                  useGrouping: false,
                              }
                          )
                        : zero.toLocaleString("en-US", {
                              minimumIntegerDigits: 2,
                              useGrouping: false,
                          })
                }
                footer="Ready for Review"
                color="#f2bc35"
            />
            <StatsCard
                icon={messaging}
                count={
                    summary?.dashboard_info?.mom_in_review
                        ? summary.dashboard_info.mom_in_review.toLocaleString(
                              "en-US",
                              {
                                  minimumIntegerDigits: 2,
                                  useGrouping: false,
                              }
                          )
                        : zero.toLocaleString("en-US", {
                              minimumIntegerDigits: 2,
                              useGrouping: false,
                          })
                }
                footer="MoMs in Review"
                color="#ff4d61"
            />
            <StatsCard
                icon={ok}
                count={
                    summary?.dashboard_info?.mom_generated
                        ? summary?.dashboard_info?.mom_generated.toLocaleString(
                              "en-US",
                              {
                                  minimumIntegerDigits: 2,
                                  useGrouping: false,
                              }
                          )
                        : zero.toLocaleString("en-US", {
                              minimumIntegerDigits: 2,
                              useGrouping: false,
                          })
                }
                footer="MoMs Generated"
                color="#3bb273"
            />
        </Box>
    );
};

export default Stats;
