import type { TimeWindow } from "tensorzero-node";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { CHART_COLORS, formatChartNumber } from "~/utils/chart";

import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
} from "~/components/ui/chart";
import type { FeedbackCountsTimeseriesData } from "./FeedbackSamplesTimeseries";

export function FeedbackCountsTimeseries({
  countsData,
  variantNames,
  timeGranularity,
}: {
  countsData: FeedbackCountsTimeseriesData[];
  variantNames: string[];
  timeGranularity: TimeWindow;
}) {
  // Convert date strings to timestamps for proper spacing
  const countsDataWithTimestamps = countsData.map((row) => ({
    ...row,
    timestamp: new Date(row.date).getTime(),
  }));

  const chartConfig: Record<string, { label: string; color: string }> =
    variantNames.reduce(
      (config, variantName, index) => ({
        ...config,
        [variantName]: {
          label: variantName,
          color: CHART_COLORS[index % CHART_COLORS.length],
        },
      }),
      {},
    );

  // Format x-axis labels based on time granularity
  const formatXAxisTick = (value: number) => {
    const date = new Date(value);
    if (timeGranularity === "hour") {
      // Show month-day and hour for hourly granularity (without year)
      return date.toISOString().slice(5, 13).replace("T", " ") + ":00";
    }
    // Show just the date for day, week, month
    return date.toISOString().slice(0, 10);
  };

  // Format tooltip labels based on time granularity
  const formatTooltipLabel = (value: number) => {
    const date = new Date(value);
    if (timeGranularity === "hour") {
      return date.toISOString().slice(0, 13).replace("T", " ") + ":00";
    }
    return date.toISOString().slice(0, 10);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cumulative Feedback Counts Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-80 w-full">
          <AreaChart accessibilityLayer data={countsDataWithTimestamps}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="timestamp"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickLine={false}
              tickMargin={10}
              axisLine={true}
              tickFormatter={formatXAxisTick}
              scale="linear"
            />
            <YAxis
              tickLine={false}
              tickMargin={10}
              axisLine={true}
              label={{
                value: "Feedback Sample Count",
                angle: -90,
                position: "insideLeft",
                style: { textAnchor: "middle" },
                offset: 10,
              }}
              tickFormatter={(value) => formatChartNumber(Number(value))}
            />
            <ChartTooltip
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;

                const total = payload.reduce(
                  (sum, entry) => sum + (Number(entry.value) || 0),
                  0,
                );

                return (
                  <div className="border-border/50 bg-background grid min-w-[8rem] items-start gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs shadow-xl">
                    <div className="font-medium">
                      {formatTooltipLabel(label)}
                    </div>
                    <div className="grid gap-1.5">
                      {payload
                        .slice()
                        .reverse()
                        .map((entry) => (
                          <div
                            key={entry.dataKey}
                            className="flex w-full flex-wrap items-center gap-2"
                          >
                            <div
                              className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
                              style={{ backgroundColor: entry.color }}
                            />
                            <div className="flex flex-1 items-center justify-between gap-2 leading-none">
                              <span className="text-muted-foreground font-mono text-xs">
                                {entry.name}
                              </span>
                              <span className="text-foreground ml-2 font-mono font-medium tabular-nums">
                                {Number(entry.value).toLocaleString()}
                              </span>
                            </div>
                          </div>
                        ))}
                      <div className="border-border/50 flex w-full flex-wrap items-center gap-2 border-t pt-1.5">
                        <div className="h-2.5 w-2.5 shrink-0" />
                        <div className="flex flex-1 items-center justify-between leading-none">
                          <span className="text-muted-foreground font-medium">
                            Total
                          </span>
                          <span className="text-foreground font-mono font-medium tabular-nums">
                            {total.toLocaleString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              }}
            />
            <ChartLegend
              content={<ChartLegendContent className="font-mono text-xs" />}
            />
            {variantNames.map((variantName) => (
              <Area
                key={variantName}
                dataKey={variantName}
                name={variantName}
                fill={chartConfig[variantName].color}
                fillOpacity={0.4}
                stroke={chartConfig[variantName].color}
                strokeWidth={0}
                stackId="1"
              />
            ))}
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
