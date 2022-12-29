import * as React from "react";
import { LineChart, Line } from "recharts";
import { useTheme } from "@material-ui/core/styles";

export default function App({ data }) {
    const theme = useTheme();
    return (
        <LineChart width={300} height={30} data={data}>
            <Line
                type="monotone"
                dataKey="talktime"
                stroke={theme.palette.primary.tertiary}
                strokeWidth={2}
            />
        </LineChart>
    );
}
