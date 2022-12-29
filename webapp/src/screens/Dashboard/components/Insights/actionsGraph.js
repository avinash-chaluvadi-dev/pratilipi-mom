import * as React from "react";
import {
    LineChart,
    Line,
    // XAxis,
    // YAxis,
    // CartesianGrid,
    // Tooltip,
    // Legend,
    // ResponsiveContainer,
} from "recharts";
import { useTheme } from "@material-ui/core/styles";

export default function App({ data }) {
    const theme = useTheme();
    return (
        // <ResponsiveContainer width="100%" height="100%">
        <LineChart width={200} height={50} data={data}>
            <Line
                type="monotone"
                dataKey="pv"
                stroke="#8884d8"
                strokeWidth={2}
            />
        </LineChart>
        // </ResponsiveContainer>
    );
}
