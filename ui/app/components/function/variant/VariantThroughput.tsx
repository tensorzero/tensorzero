import type { VariantThroughput } from "~/utils/clickhouse/function";
import type { TimeWindow } from "~/types/tensorzero";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { CHART_COLORS, formatChartNumber } from "~/utils/chart";

import { Card, CardContent, CardHeader } from "~/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
} from "~/components/ui/chart";
import { TimeGranularitySelector } from "./TimeGranularitySelector";

export function VariantThroughput({
  variant_throughput,
  time_granularity,
  onTimeGranularityChange,
}: {
  variant_throughput: VariantThroughput[];
  time_granularity: TimeWindow;
  onTimeGranularityChange: (time_granularity: TimeWindow) => void;
}) {
  const { data, variantNames } = transformVariantThroughput(variant_throughput);

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
                  value: "Inference Count",
                  angle: -90,
                  position: "insideLeft",
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

export type VariantThroughputData = {
  date: string;
  [key: string]: string | number;
};

export function transformVariantThroughput(parsedRows: VariantThroughput[]): {
  data: VariantThroughputData[];
  variantNames: string[];
} {
  const variantNames = [...new Set(parsedRows.map((row) => row.variant_name))];

  // Group by date
  const groupedByDate = parsedRows.reduce<
    Record<string, Record<string, number>>
  >((acc, row) => {
    const { period_start, variant_name, count } = row;

    if (!acc[period_start]) {
      acc[period_start] = {};
    }

    acc[period_start][variant_name] = count;
    return acc;
  }, {});

  // Convert to array and sort by date
  const data = Object.entries(groupedByDate)
    .map(([date, variants]) => {
      const row: VariantThroughputData = { date };
      variantNames.forEach((variant) => {
        row[variant] = variants[variant] ?? 0;
      });
      return row;
    })
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    .slice(-10); // Take only the 10 most recent periods

  return {
    data,
    variantNames,
  };
}
