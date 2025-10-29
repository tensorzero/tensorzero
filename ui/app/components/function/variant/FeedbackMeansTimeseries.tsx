import type { TimeWindow } from "~/types/tensorzero";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
} from "recharts";
import { type ReactNode } from "react";
import {
  CHART_COLORS,
  formatDetailedNumber,
  formatXAxisTimestamp,
  formatTooltipTimestamp,
} from "~/utils/chart";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { TimeGranularitySelector } from "./TimeGranularitySelector";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
} from "~/components/ui/chart";
import type {
  FeedbackMeansTimeseriesData,
  FeedbackCountsTimeseriesData,
} from "./FeedbackSamplesTimeseries";

export function FeedbackMeansTimeseries({
  meansData,
  countsData,
  variantNames,
  timeGranularity,
  metricName,
  time_granularity,
  onTimeGranularityChange,
}: {
  meansData: FeedbackMeansTimeseriesData[];
  countsData: FeedbackCountsTimeseriesData[];
  variantNames: string[];
  timeGranularity: TimeWindow;
  metricName: string;
  time_granularity: TimeWindow;
  onTimeGranularityChange: (value: TimeWindow) => void;
}) {
  const meanDataWithTimestamps: Array<
    FeedbackMeansTimeseriesData & { timestamp: number }
  > = meansData.map((row) => ({
    ...row,
    timestamp: new Date(row.date).getTime(),
  }));

  // Create a mapping from date to counts for tooltip
  const countsDataByDate = countsData.reduce<
    Record<string, FeedbackCountsTimeseriesData>
  >((acc, row) => {
    acc[row.date] = row;
    return acc;
  }, {});

  const getMeanValue = (
    row: (typeof meanDataWithTimestamps)[number],
    variant: string,
  ): number | null | undefined =>
    row[variant as keyof typeof row] as number | null | undefined;

  const getLowerValue = (
    row: (typeof meanDataWithTimestamps)[number],
    variant: string,
  ): number | null | undefined =>
    row[`${variant}_cs_lower` as keyof typeof row] as number | null | undefined;

  const getUpperValue = (
    row: (typeof meanDataWithTimestamps)[number],
    variant: string,
  ): number | null | undefined =>
    row[`${variant}_cs_upper` as keyof typeof row] as number | null | undefined;

  const meanDataWithValues = meanDataWithTimestamps.filter((row) =>
    variantNames.some((variant) => getMeanValue(row, variant) !== null),
  );

  const meanChartData =
    meanDataWithValues.length > 0 ? meanDataWithValues : meanDataWithTimestamps;

  const variantsWithMeanData = variantNames.filter((variant) =>
    meanChartData.some((row) => getMeanValue(row, variant) !== null),
  );

  const variantsWithConfidence = variantsWithMeanData.filter((variant) =>
    meanChartData.some(
      (row) =>
        getLowerValue(row, variant) !== null &&
        getUpperValue(row, variant) !== null,
    ),
  );

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

  const meanChartConfig: Record<string, { label: string; color: string }> =
    variantsWithMeanData.reduce(
      (config, variantName) => ({
        ...config,
        [variantName]: chartConfig[variantName],
      }),
      {},
    );

  const meanContainerConfig =
    Object.keys(meanChartConfig).length > 0 ? meanChartConfig : chartConfig;

  const meanLegendPayload =
    variantsWithMeanData.length > 0
      ? variantsWithMeanData.map((variantName) => ({
          value: variantName,
          type: "line" as const,
          color: chartConfig[variantName].color,
          dataKey: variantName,
        }))
      : variantNames.map((variantName) => ({
          value: variantName,
          type: "line" as const,
          color: chartConfig[variantName].color,
          dataKey: variantName,
        }));

  // Format x-axis labels based on time granularity
  const formatXAxisTick = (value: number) =>
    formatXAxisTimestamp(new Date(value), timeGranularity);

  // Format tooltip labels based on time granularity
  const formatTooltipLabel = (value: number) =>
    formatTooltipTimestamp(new Date(value), timeGranularity);

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-4">
        <div className="space-y-1.5">
          <CardTitle>
            Estimated Performance:{" "}
            <span className="font-mono font-semibold">{metricName}</span>
          </CardTitle>
          <CardDescription>
            This chart displays the cumulative mean score of each variant. The
            shaded areas indicate 95% confidence. The snapshots depict how the
            estimates are converging over time.
          </CardDescription>
        </div>
        <TimeGranularitySelector
          time_granularity={time_granularity}
          onTimeGranularityChange={onTimeGranularityChange}
          includeCumulative={false}
          includeMinute={true}
          includeHour={true}
        />
      </CardHeader>
      <CardContent>
        <ChartContainer config={meanContainerConfig} className="h-80 w-full">
          <ComposedChart accessibilityLayer data={meanChartData}>
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
                value: "Estimated Performance",
                angle: -90,
                position: "insideLeft",
                style: { textAnchor: "middle" },
                offset: 10,
              }}
              tickFormatter={(value) => {
                const num = Number(value);
                return formatDetailedNumber(Number.isFinite(num) ? num : 0);
              }}
            />
            <ChartTooltip
              content={({ active, label, payload }) => {
                if (!active || !payload?.length) return null;

                const dataPoint = payload[0]?.payload;
                if (!dataPoint) return null;

                // Collect variant data and sort by mean descending
                const variantData: Array<{
                  name: string;
                  mean: number;
                  lower: number | null | undefined;
                  upper: number | null | undefined;
                  count: number;
                }> = [];

                variantsWithMeanData.forEach((variantName) => {
                  const mean = dataPoint[variantName];
                  if (mean === null || mean === undefined) {
                    return;
                  }

                  const lower = dataPoint[`${variantName}_cs_lower`] as
                    | number
                    | null
                    | undefined;
                  const upper = dataPoint[`${variantName}_cs_upper`] as
                    | number
                    | null
                    | undefined;

                  // Get count from counts data
                  const dateStr = dataPoint.date as string;
                  const count =
                    (countsDataByDate[dateStr]?.[variantName] as
                      | number
                      | undefined) ?? 0;

                  variantData.push({
                    name: variantName,
                    mean,
                    lower,
                    upper,
                    count,
                  });
                });

                // Sort by mean descending
                variantData.sort((a, b) => b.mean - a.mean);

                const rows: ReactNode[] = variantData.map(
                  ({ name, mean, lower, upper, count }) => {
                    const hasBounds =
                      lower !== null &&
                      lower !== undefined &&
                      upper !== null &&
                      upper !== undefined;

                    return (
                      <div
                        key={name}
                        className="flex w-full items-center gap-2"
                      >
                        <div
                          className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
                          style={{
                            backgroundColor: chartConfig[name].color,
                          }}
                        />
                        <span className="text-muted-foreground font-mono text-xs">
                          {name}
                        </span>
                        <div className="flex items-center gap-1.5 font-mono tabular-nums">
                          <span className="text-foreground font-medium">
                            {formatDetailedNumber(mean)}
                          </span>
                          {hasBounds && (
                            <span className="text-muted-foreground text-[10px]">
                              ({formatDetailedNumber(lower as number)},{" "}
                              {formatDetailedNumber(upper as number)})
                            </span>
                          )}
                          <span className="text-muted-foreground ml-3 text-[10px]">
                            n={count.toLocaleString()}
                          </span>
                        </div>
                      </div>
                    );
                  },
                );

                if (!rows.length) return null;

                return (
                  <div className="border-border/50 bg-background grid min-w-[8rem] items-start gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs shadow-xl">
                    <div className="font-medium">
                      {formatTooltipLabel(label)}
                    </div>
                    <div className="grid gap-1.5">{rows}</div>
                  </div>
                );
              }}
            />
            <ChartLegend
              payload={meanLegendPayload}
              content={<ChartLegendContent className="font-mono text-xs" />}
            />
            {variantsWithMeanData.flatMap((variantName) => {
              const color = chartConfig[variantName].color;
              const meanKey = variantName;
              const lowerKey = `${variantName}_cs_lower`;
              const widthKey = `${variantName}_cs_width`;
              const hasConfidence =
                variantsWithConfidence.includes(variantName);

              const elements: ReactNode[] = [];

              if (hasConfidence) {
                elements.push(
                  <Area
                    key={`${variantName}-cs-lower`}
                    type="monotone"
                    dataKey={lowerKey}
                    stackId={variantName}
                    stroke="transparent"
                    fill="transparent"
                    isAnimationActive={false}
                    connectNulls
                  />,
                  <Area
                    key={`${variantName}-cs-band`}
                    type="monotone"
                    dataKey={widthKey}
                    stackId={variantName}
                    stroke="transparent"
                    fill={color}
                    fillOpacity={0.25}
                    isAnimationActive={false}
                    connectNulls
                    legendType="none"
                  />,
                );
              }

              elements.push(
                <Line
                  key={`${variantName}-mean`}
                  type="monotone"
                  dataKey={meanKey}
                  name={variantName}
                  stroke={color}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 4 }}
                  isAnimationActive={false}
                  connectNulls
                />,
              );

              return elements;
            })}
          </ComposedChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
