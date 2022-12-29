import React, { Component } from "react";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    Label,
    CartesianGrid,
} from "recharts";

class BarGraph extends Component {
    constructor(props) {
        super(props);
        this.selectBar = this.selectBar.bind(this);

        this.state = {
            title: this.props.title,
            dataKey: this.props.data.dataKey,
            oxLabel: this.props.data.oxLabel,
            oyLabel: this.props.data.oyLabel,
            values: this.props.data.values,
            yLimit: this.props.data.yLimit,
            labels: this.props.labels,
        };
    }

    selectBar(event) {
        let updatedLabels = [];
        for (let i = 0; i < this.state.labels.length; i++) {
            let label = this.state.labels[i];
            if (label.key !== event.dataKey) {
                updatedLabels.push(label);
            } else {
                if (/\s/.test(label.key)) {
                    let newLabel = {
                        key: label.key.trim(),
                        color: label.color,
                    };
                    updatedLabels.push(newLabel);
                } else {
                    let newLabel = { key: label.key + " ", color: label.color };
                    updatedLabels.push(newLabel);
                }
            }
        }
        this.setState({
            labels: updatedLabels,
        });
    }

    render() {
        return (
            <div>
                {/* <h3>{this.props.title}</h3> */}
                <BarChart
                    width={1250}
                    height={420}
                    data={this.state.values}
                    margin={{ top: 30, right: 20, left: 50, bottom: 1 }}
                >
                    <CartesianGrid strokeDasharray="3 3" />

                    <XAxis dataKey={this.state.dataKey}>
                        <Label
                            value={this.state.oxLabel}
                            position="insideBottomRight"
                            dy={10}
                            dx={20}
                        />
                    </XAxis>
                    <YAxis type="number" domain={this.state.yLimit}>
                        <Label
                            value={this.state.oyLabel}
                            position="left"
                            angle={-90}
                            dy={-20}
                            dx={-10}
                        />
                    </YAxis>
                    <Tooltip />
                    <Legend
                        layout="horizontal"
                        verticalAlign="top"
                        height={72}
                        align="left"
                        wrapperStylestyle={{ margin: 10 }}
                        onClick={this.selectBar}
                    />

                    {this.state.labels.map((label, index) => (
                        <Bar
                            key={index}
                            dataKey={label.key}
                            fill={label.color}
                            stackId={this.state.dataKey}
                            radius={
                                this.state.labels.length === index + 1
                                    ? [10, 10, 0, 0]
                                    : {}
                            }
                            barSize={15}
                        />
                    ))}
                </BarChart>
                <h3>{this.props.title}</h3>
            </div>
        );
    }
}

export default BarGraph;
