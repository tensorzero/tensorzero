import type {
  CumulativeFeedbackTimeSeriesPoint,
  TimeWindow,
} from "tensorzero-node";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
} from "recharts";
import { type ReactNode } from "react";
import {
  CHART_COLORS,
  formatChartNumber,
  formatDetailedNumber,
} from "~/utils/chart";
import { addPeriod, normalizePeriod } from "~/utils/date";

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
  onCumulativeFeedbackTimeGranularityChange: onTimeGranularityChange,
}: {
  feedbackTimeseries: CumulativeFeedbackTimeSeriesPoint[];
  time_granularity: TimeWindow;
  onCumulativeFeedbackTimeGranularityChange: (
    time_granularity: TimeWindow,
  ) => void;
}) {
  const { countsData, meansData, variantNames } = transformFeedbackTimeseries(
    feedbackTimeseries,
    time_granularity,
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
    </div>
  );
}

type FeedbackTimeseriesPointByVariant = {
  count: number;
  mean: number | null;
  cs_lower: number | null;
  cs_upper: number | null;
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
    const { period_end, variant_name, count, mean, cs_lower, cs_upper } = row;

    if (!acc[period_end]) {
      acc[period_end] = {};
    }

    const sanitizedCount = Number(count);
    const sanitizedMean =
      mean === null || mean === undefined ? null : Number(mean);
    const sanitizedLower =
      cs_lower === null || cs_lower === undefined ? null : Number(cs_lower);
    const sanitizedUpper =
      cs_upper === null || cs_upper === undefined ? null : Number(cs_upper);

    acc[period_end][variant_name] = {
      count: Number.isFinite(sanitizedCount) ? sanitizedCount : 0,
      mean:
        sanitizedMean !== null && Number.isFinite(sanitizedMean)
          ? sanitizedMean
          : null,
      cs_lower:
        sanitizedLower !== null && Number.isFinite(sanitizedLower)
          ? sanitizedLower
          : null,
      cs_upper:
        sanitizedUpper !== null && Number.isFinite(sanitizedUpper)
          ? sanitizedUpper
          : null,
    };
    return acc;
  }, {});

  // Get all unique periods from the data, sorted chronologically
  const allPeriods = Object.keys(groupedByDate).sort();

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
        const nextPeriodStr = addPeriod(current, timeGranularity);
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
      const normalized = normalizePeriod(period, timeGranularity);
      additionalPeriods.unshift(normalized.toISOString());
    }

    filledPeriods.unshift(...additionalPeriods);
  }

  // Take only the last 10 periods
  const periodsToShow = filledPeriods.slice(-10);

  // Initialize forward-filled stats
  const lastKnownCounts: Record<string, number> = {};
  const lastKnownMeans: Record<string, number | null> = {};
  const lastKnownLower: Record<string, number | null> = {};
  const lastKnownUpper: Record<string, number | null> = {};
  variantNames.forEach((variant) => {
    lastKnownCounts[variant] = 0;
    lastKnownMeans[variant] = null;
    lastKnownLower[variant] = null;
    lastKnownUpper[variant] = null;
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

        if (
          periodData.cs_lower !== null &&
          periodData.cs_upper !== null &&
          Number.isFinite(periodData.cs_lower) &&
          Number.isFinite(periodData.cs_upper)
        ) {
          lastKnownLower[variant] = periodData.cs_lower;
          lastKnownUpper[variant] = periodData.cs_upper;
        } else {
          lastKnownLower[variant] = null;
          lastKnownUpper[variant] = null;
        }
      }

      countsRow[variant] = lastKnownCounts[variant];

      const mean = lastKnownMeans[variant];
      const lower = lastKnownLower[variant];
      const upper = lastKnownUpper[variant];
      const width = lower !== null && upper !== null ? upper - lower : null;

      meansRow[variant] = mean ?? null;
      meansRow[`${variant}_cs_lower`] = lower ?? null;
      meansRow[`${variant}_cs_upper`] = upper ?? null;
      meansRow[`${variant}_cs_width`] = width ?? null;
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
