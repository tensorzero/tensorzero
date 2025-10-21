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
  const { data, variantNames } =
    transformFeedbackTimeseries(feedbackTimeseries);

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
            <AreaChart accessibilityLayer data={data}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="date"
                tickLine={false}
                tickMargin={10}
                axisLine={true}
                tickFormatter={(value) =>
                  new Date(value).toISOString().slice(0, 10)
                }
              />
              <YAxis
                tickLine={false}
                tickMargin={10}
                axisLine={true}
                label={{
                  value: "Feedback Sample Count",
                  angle: -90,
                  position: "insideLeft",
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
                        {new Date(label).toISOString().slice(0, 10)}
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
): {
  data: FeedbackTimeseriesData[];
  variantNames: string[];
} {
  const variantNames = [...new Set(parsedRows.map((row) => row.variant_name))];

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

  // Convert to array and sort by date
  // Note: ClickHouse already returns cumulative counts, so we don't need to compute them again
  const sortedData = Object.entries(groupedByDate)
    .map(([date, variants]) => {
      const row: FeedbackTimeseriesData = { date };
      variantNames.forEach((variant) => {
        row[variant] = variants[variant] ?? 0;
      });
      return row;
    })
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  // Forward-fill missing cumulative values for each variant
  // (ClickHouse only returns a row for a variant when it has new data in that period)
  const lastKnownCounts: Record<string, number> = {};
  variantNames.forEach((variant) => {
    lastKnownCounts[variant] = 0;
  });

  const filledData = sortedData.map((row) => {
    const filledRow: FeedbackTimeseriesData = { date: row.date };
    variantNames.forEach((variant) => {
      const currentValue = row[variant] as number;
      // If we have a new value, use it and update the last known count
      if (currentValue > 0) {
        lastKnownCounts[variant] = currentValue;
      }
      // Always use the last known count (forward-fill)
      filledRow[variant] = lastKnownCounts[variant];
    });
    return filledRow;
  });

  // Take only the 10 most recent periods
  const data = filledData.slice(-10);

  return {
    data,
    variantNames,
  };
}
