import type {
  CumulativeFeedbackTimeSeriesPoint,
  TimeWindow,
} from "tensorzero-node";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { CHART_COLORS } from "~/utils/chart";

import { Card, CardContent, CardHeader } from "~/components/ui/card";
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
        <CardHeader className="flex flex-row items-start justify-between">
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
    </div>
  );
}

export type FeedbackTimeseriesData = {
  date: string;
  [key: string]: string | number;
};

export function transformFeedbackTimeseries(
  parsedRows: CumulativeFeedbackTimeSeriesPoint[],
  timeGranularity: TimeWindow,
): {
  data: FeedbackTimeseriesData[];
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

  // Initialize cumulative counts for forward-filling
  const lastKnownCounts: Record<string, number> = {};
  variantNames.forEach((variant) => {
    lastKnownCounts[variant] = 0;
  });

  // Create data for each period, forward-filling cumulative counts
  const data: FeedbackTimeseriesData[] = periodsToShow.map((period) => {
    const row: FeedbackTimeseriesData = { date: period };

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
