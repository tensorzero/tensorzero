import type {
  CumulativeFeedbackTimeSeriesPoint,
  TimeWindow,
} from "~/types/tensorzero";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { CHART_COLORS, formatChartNumber } from "~/utils/chart";
import { normalizePeriod, addPeriod } from "~/utils/date";

import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
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
  const { data, variantNames } = transformFeedbackTimeseries(
    feedbackTimeseries,
    time_granularity,
  );

  // Convert date strings to timestamps for proper spacing
  const dataWithTimestamps = data.map((row) => ({
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
            <AreaChart accessibilityLayer data={dataWithTimestamps}>
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
    </div>
  );
}

type FeedbackTimeSeriesData = {
  date: string;
  [key: string]: string | number;
};

/**
 * Transforms cumulative feedback time series data for chart visualization.
 *
 * This function processes raw feedback data points and prepares them for display
 * in a time series chart by:
 * 1. Grouping data by time period and variant
 * 2. Filling in missing periods to ensure continuous time series (because ClickHouse
 *    query returns sparse data)
 * 3. Forward-filling cumulative counts (so gaps maintain the last known value)
 * 4. Limiting to the most recent 10 periods for display
 *
 * The function ensures that cumulative counts are preserved across periods where
 * no new feedback was recorded, which is essential for displaying cumulative metrics.
 *
 * @param parsedRows - Array of cumulative feedback data points from ClickHouse
 * @param timeGranularity - The time window unit (minute, hour, day, week, month, cumulative)
 * @returns Object containing:
 *   - data: Array of chart-ready data points with period dates and variant counts
 *   - variantNames: Unique list of variant names present in the data
 */
function transformFeedbackTimeseries(
  parsedRows: CumulativeFeedbackTimeSeriesPoint[],
  timeGranularity: TimeWindow,
): {
  data: FeedbackTimeSeriesData[];
  variantNames: string[];
} {
  const variantNames = [...new Set(parsedRows.map((row) => row.variant_name))];

  // If no data, return empty
  if (parsedRows.length === 0) {
    return { data: [], variantNames: [] };
  }

  // Group by date
  const groupedByDate = parsedRows.reduce<
    Record<string, Record<string, number>>
  >((acc, row) => {
    const { period_end, variant_name, count } = row;

    if (!acc[period_end]) {
      acc[period_end] = {};
    }

    // Convert bigint to number
    acc[period_end][variant_name] = Number(count);
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

  // Initialize cumulative counts for forward-filling
  const lastKnownCounts: Record<string, number> = {};
  variantNames.forEach((variant) => {
    lastKnownCounts[variant] = 0;
  });

  // Create data for each period, forward-filling cumulative counts
  const data: FeedbackTimeSeriesData[] = periodsToShow.map((period) => {
    const row: FeedbackTimeSeriesData = { date: period };

    variantNames.forEach((variant) => {
      // Check if we have actual data for this period
      const periodData = groupedByDate[period];
      if (periodData && periodData[variant] !== undefined) {
        // Update with actual cumulative count from ClickHouse
        lastKnownCounts[variant] = periodData[variant];
      }
      // Use the last known cumulative count (forward-fill)
      row[variant] = lastKnownCounts[variant];
    });

    return row;
  });

  return {
    data,
    variantNames,
  };
}
