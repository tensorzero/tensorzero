import type {
  TimeWindowUnit,
  VariantThroughput,
} from "~/utils/clickhouse/function";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";

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
  ChartTooltipContent,
} from "~/components/ui/chart";
import { TimeGranularitySelector } from "./TimeGranularitySelector";

const CHART_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
] as const;

export function VariantThroughput({
  variant_throughput,
  time_granularity,
  onTimeGranularityChange,
}: {
  variant_throughput: VariantThroughput[];
  time_granularity: TimeWindowUnit;
  onTimeGranularityChange: (time_granularity: TimeWindowUnit) => void;
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
          <div>
            <CardTitle>Throughput Over Time</CardTitle>
            <CardDescription>
              <span>Showing inference counts by variant over time</span>
            </CardDescription>
          </div>
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
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis tickLine={false} tickMargin={10} axisLine={true} />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(label) =>
                      new Date(label).toLocaleDateString()
                    }
                    formatter={(value, name) => {
                      return (
                        <div className="flex flex-1 items-center justify-between leading-none">
                          <span className="text-muted-foreground">{name}</span>
                          <span className="text-foreground font-mono font-medium tabular-nums">
                            {value.toLocaleString()}
                          </span>
                        </div>
                      );
                    }}
                  />
                }
              />
              <ChartLegend content={<ChartLegendContent />} />
              {variantNames.map((variantName) => (
                <Area
                  key={variantName}
                  dataKey={variantName}
                  name={variantName}
                  fill={chartConfig[variantName].color}
                  fillOpacity={0.4}
                  stroke={chartConfig[variantName].color}
                  strokeWidth={2}
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
