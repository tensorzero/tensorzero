import type {
  CumulativeFeedbackTimeSeriesPoint,
  TimeWindow,
} from "tensorzero-node";
import {
  Area,
  AreaChart,
  CartesianGrid,
  LineChart,
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
import { TimeGranularitySelector } from "./TimeGranularitySelector";

export function FeedbackSamplesTimeseries({
  feedbackTimeseries,
  time_granularity,
  onTimeGranularityChange,
}: {
  feedbackTimeseries: CumulativeFeedbackTimeSeriesPoint[];
  time_granularity: TimeWindow;
  onTimeGranularityChange: (time_granularity: TimeWindow) => void;
}) {
  const { countsData, meansData, variantNames } = transformFeedbackTimeseries(
    feedbackTimeseries,
    time_granularity,
  );

  console.log(
    "meansData from transform:",
    meansData.map((d) => d.date),
  );

  // Convert date strings to timestamps for proper spacing
  const countsDataWithTimestamps = countsData.map((row) => ({
    ...row,
    timestamp: new Date(row.date).getTime(),
  }));

  const meanDataWithTimestamps: Array<
    FeedbackMeansTimeseriesData & { timestamp: number }
  > = meansData.map((row) => ({
    ...row,
    timestamp: new Date(row.date).getTime(),
  }));

  console.log(
    "meanDataWithTimestamps:",
    meanDataWithTimestamps.map((d) => ({
      date: d.date,
      timestamp: d.timestamp,
    })),
  );

  // Filter to only include data points where at least one variant has a non-null value
  // This is necessary because Recharts won't render lines if the first point is null
  const meanChartData = meanDataWithTimestamps.filter((row) =>
    variantNames.some(
      (variant) =>
        row[variant as keyof typeof row] !== null &&
        row[variant as keyof typeof row] !== undefined,
    ),
  );

  console.log("variantNames:", variantNames);
  console.log("Full meanChartData:", meanChartData);
  console.log("meanChartData length:", meanChartData.length);

  // Check if we have any non-null values at all
  variantNames.forEach((variant) => {
    const values = meanChartData
      .map((d) => d[variant as keyof typeof d])
      .filter(
        (v) => v !== null && v !== undefined && typeof v === "number",
      ) as number[];
    console.log(
      `${variant}: ${values.length} non-null values, range: ${Math.min(...values)} to ${Math.max(...values)}`,
    );
  });

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
    if (time_granularity === "hour") {
      // Show month-day and hour for hourly granularity (without year)
      return date.toISOString().slice(5, 13).replace("T", " ") + ":00";
    }
    // Show just the date for day, week, month
    return date.toISOString().slice(0, 10);
  };

  // Format tooltip labels based on time granularity
  const formatTooltipLabel = (value: number) => {
    const date = new Date(value);
    if (time_granularity === "hour") {
      return date.toISOString().slice(0, 13).replace("T", " ") + ":00";
    }
    return date.toISOString().slice(0, 10);
  };

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Cumulative Feedback Counts Over Time</CardTitle>
          <TimeGranularitySelector
            time_granularity={time_granularity}
            onTimeGranularityChange={onTimeGranularityChange}
            includeCumulative={false}
            includeMinute={false}
            includeHour={true}
          />
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
                tickFormatter={(value) => {
                  const num = Number(value);
                  if (num >= 1000000) {
                    return (num / 1000000).toFixed(1) + "M";
                  } else if (num >= 1000) {
                    return (num / 1000).toFixed(1) + "K";
                  }
                  return num.toString();
                }}
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
      <Card>
        <CardHeader>
          <CardTitle>Mean Reward Confidence Sequences</CardTitle>
          <CardDescription>
            Showing per-variant mean reward estimates with confidence sequences.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-80 w-full">
            <LineChart accessibilityLayer data={meanChartData}>
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

                  variantNames.forEach((variantName) => {
                    const mean = dataPoint[variantName];
                    if (mean === null || mean === undefined) {
                      return;
                    }

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
                          {mean !== null ? (
                            <span className="text-foreground font-mono font-medium tabular-nums">
                              {formatDetailedNumber(mean)}
                            </span>
                          ) : (
                            <span className="text-muted-foreground text-[10px]">
                              Mean pending
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
                content={<ChartLegendContent className="font-mono text-xs" />}
              />
              {variantNames.map((variantName) => {
                console.log(
                  "Creating Line for:",
                  variantName,
                  "with dataKey:",
                  variantName,
                  "stroke:",
                  chartConfig[variantName].color,
                );

                return (
                  <Line
                    key={variantName}
                    type="monotone"
                    dataKey={variantName}
                    name={variantName}
                    stroke={chartConfig[variantName].color}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                );
              })}
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  );
}

type FeedbackTimeseriesPointByVariant = {
  count: number;
  mean: number | null;
};

export type FeedbackCountsTimeseriesData = {
  date: string;
  [key: string]: string | number;
};

export type FeedbackMeansTimeseriesData = {
  date: string;
  [key: string]: string | number | null;
};

export function transformFeedbackTimeseries(
  parsedRows: CumulativeFeedbackTimeSeriesPoint[],
  timeGranularity: TimeWindow,
): {
  countsData: FeedbackCountsTimeseriesData[];
  meansData: FeedbackMeansTimeseriesData[];
  variantNames: string[];
} {
  const variantNames = [...new Set(parsedRows.map((row) => row.variant_name))];

  // If no data, return empty
  if (parsedRows.length === 0) {
    return { countsData: [], meansData: [], variantNames: [] };
  }

  // Group by date
  const groupedByDate = parsedRows.reduce<
    Record<string, Record<string, FeedbackTimeseriesPointByVariant>>
  >((acc, row) => {
    const { period_end, variant_name, count, mean } = row;

    if (!acc[period_end]) {
      acc[period_end] = {};
    }

    const sanitizedCount = Number(count);
    const sanitizedMean =
      mean === null || mean === undefined ? null : Number(mean);

    acc[period_end][variant_name] = {
      count: Number.isFinite(sanitizedCount) ? sanitizedCount : 0,
      mean:
        sanitizedMean !== null && Number.isFinite(sanitizedMean)
          ? sanitizedMean
          : null,
    };
    return acc;
  }, {});

  // Get all unique periods from the data, sorted chronologically
  const allPeriods = Object.keys(groupedByDate).sort();

  // Helper to normalize a date to match ClickHouse's period format
  const normalizePeriod = (date: Date): Date => {
    const normalized = new Date(date);
    switch (timeGranularity) {
      case "minute":
        // Truncate to minute
        normalized.setUTCSeconds(0, 0);
        break;
      case "hour":
        // Truncate to hour
        normalized.setUTCMinutes(0, 0, 0);
        break;
      case "day":
        // Truncate to day
        normalized.setUTCHours(0, 0, 0, 0);
        break;
      case "week":
        // Truncate to day
        normalized.setUTCHours(0, 0, 0, 0);
        break;
      case "month":
        // Truncate to day
        normalized.setUTCHours(0, 0, 0, 0);
        break;
      case "cumulative":
        // No truncation needed for cumulative
        break;
    }
    return normalized;
  };

  // Helper to add one period to a date based on granularity
  const addPeriod = (date: Date): string => {
    const result = new Date(date);
    switch (timeGranularity) {
      case "minute":
        result.setUTCMinutes(result.getUTCMinutes() + 1);
        break;
      case "hour":
        result.setUTCHours(result.getUTCHours() + 1);
        break;
      case "day":
        result.setUTCDate(result.getUTCDate() + 1);
        break;
      case "week":
        result.setUTCDate(result.getUTCDate() + 7);
        break;
      case "month":
        result.setUTCMonth(result.getUTCMonth() + 1);
        break;
      case "cumulative":
        // No period addition for cumulative
        break;
    }
    return normalizePeriod(result).toISOString();
  };

  // Fill in missing periods between the ones we have from ClickHouse
  const filledPeriods: string[] = [];
  for (let i = 0; i < allPeriods.length; i++) {
    const currentPeriod = allPeriods[i];
    filledPeriods.push(currentPeriod);

    // Check if there's a next period to compare against
    if (i < allPeriods.length - 1) {
      const nextPeriod = allPeriods[i + 1];
      let current = new Date(currentPeriod);
      const next = new Date(nextPeriod);

      // Fill in any missing periods between current and next
      while (true) {
        const nextPeriodStr = addPeriod(current);
        current = new Date(nextPeriodStr);
        if (current.getTime() >= next.getTime()) break;
        filledPeriods.push(nextPeriodStr);
      }
    }
  }

  // If we have fewer than 10 periods, add more going backwards from the earliest
  if (filledPeriods.length < 10) {
    const earliestPeriod = new Date(filledPeriods[0]);
    const periodsToAdd = 10 - filledPeriods.length;
    const additionalPeriods: string[] = [];

    for (let i = 1; i <= periodsToAdd; i++) {
      const period = new Date(earliestPeriod);
      switch (timeGranularity) {
        case "minute":
          period.setUTCMinutes(earliestPeriod.getUTCMinutes() - i);
          break;
        case "hour":
          period.setUTCHours(earliestPeriod.getUTCHours() - i);
          break;
        case "day":
          period.setUTCDate(earliestPeriod.getUTCDate() - i);
          break;
        case "week":
          period.setUTCDate(earliestPeriod.getUTCDate() - i * 7);
          break;
        case "month":
          period.setUTCMonth(earliestPeriod.getUTCMonth() - i);
          break;
        case "cumulative":
          // No period subtraction for cumulative
          break;
      }
      const normalized = normalizePeriod(period);
      additionalPeriods.unshift(normalized.toISOString());
    }

    filledPeriods.unshift(...additionalPeriods);
  }

  // Take only the last 10 periods
  const periodsToShow = filledPeriods.slice(-10);

  // Initialize forward-filled stats
  const lastKnownCounts: Record<string, number> = {};
  const lastKnownMeans: Record<string, number | null> = {};
  variantNames.forEach((variant) => {
    lastKnownCounts[variant] = 0;
    lastKnownMeans[variant] = null;
  });

  const countsData: FeedbackCountsTimeseriesData[] = [];
  const meansData: FeedbackMeansTimeseriesData[] = [];

  periodsToShow.forEach((period) => {
    const countsRow: FeedbackCountsTimeseriesData = { date: period };
    const meansRow: FeedbackMeansTimeseriesData = { date: period };

    variantNames.forEach((variant) => {
      const periodData = groupedByDate[period]?.[variant];

      if (periodData) {
        if (Number.isFinite(periodData.count)) {
          lastKnownCounts[variant] = periodData.count;
        }

        if (periodData.mean !== null) {
          lastKnownMeans[variant] = periodData.mean;
        }
      }

      countsRow[variant] = lastKnownCounts[variant];

      const mean = lastKnownMeans[variant];

      meansRow[variant] = mean ?? null;
    });

    countsData.push(countsRow);
    meansData.push(meansRow);
  });

  return {
    countsData,
    meansData,
    variantNames,
  };
}
