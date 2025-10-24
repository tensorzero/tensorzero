import type { TimeWindow } from "tensorzero-node";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
} from "recharts";
import { type ReactNode } from "react";
import { CHART_COLORS, formatDetailedNumber } from "~/utils/chart";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
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
}: {
  meansData: FeedbackMeansTimeseriesData[];
  countsData: FeedbackCountsTimeseriesData[];
  variantNames: string[];
  timeGranularity: TimeWindow;
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
        <CardTitle>Mean Reward Over Time</CardTitle>
        <CardDescription>
          Per-variant mean rewards, with confidence sequences when available.
        </CardDescription>
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
                value: "Mean Reward",
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

                const rows: ReactNode[] = [];

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
                  const hasBounds =
                    lower !== null &&
                    lower !== undefined &&
                    upper !== null &&
                    upper !== undefined;

                  // Get count from counts data
                  const dateStr = dataPoint.date as string;
                  const count =
                    (countsDataByDate[dateStr]?.[variantName] as
                      | number
                      | undefined) ?? 0;

                  rows.push(
                    <div
                      key={variantName}
                      className="flex w-full flex-wrap items-start gap-2"
                    >
                      <div
                        className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
                        style={{
                          backgroundColor: chartConfig[variantName].color,
                        }}
                      />
                      <div className="flex flex-1 flex-col gap-0.5">
                        <span className="text-muted-foreground font-mono text-xs">
                          {variantName}
                        </span>
                        <span className="text-foreground font-mono font-medium tabular-nums">
                          {formatDetailedNumber(mean)}
                        </span>
                        <span className="text-muted-foreground font-mono text-[10px]">
                          n = {count.toLocaleString()}
                        </span>
                        {hasBounds ? (
                          <span className="text-muted-foreground font-mono text-[10px] tabular-nums">
                            {formatDetailedNumber(lower as number)} –{" "}
                            {formatDetailedNumber(upper as number)}
                          </span>
                        ) : (
                          <span className="text-muted-foreground text-[10px]">
                            Confidence pending
                          </span>
                        )}
                      </div>
                    </div>,
                  );
                });

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
