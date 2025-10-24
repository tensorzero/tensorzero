import type { VariantPerformanceRow } from "~/utils/clickhouse/function";
// import { TrendingUp } from "lucide-react";
import { Bar, BarChart, ErrorBar, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  formatChartNumber,
  formatDetailedNumber,
  formatXAxisTimestamp,
  formatTooltipTimestamp,
  CHART_COLORS,
} from "~/utils/chart";

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
import { useTimeGranularityParam } from "~/hooks/use-time-granularity-param";

export function VariantPerformance({
  variant_performances,
  metric_name,
  singleVariantMode = false,
}: {
  variant_performances: VariantPerformanceRow[];
  metric_name: string;
  singleVariantMode?: boolean;
}) {
  const [time_granularity, onTimeGranularityChange] = useTimeGranularityParam(
    "time_granularity",
    "week",
  );
  const { data, variantNames } =
    transformVariantPerformances(variant_performances);

  const chartConfig: Record<string, { label: string; color: string }> =
    variantNames.reduce(
      (config, variantName, index) => ({
        ...config,
        [variantName]: {
          label: variantName,
          color: singleVariantMode
            ? CHART_COLORS[0]
            : CHART_COLORS[index % CHART_COLORS.length],
        },
      }),
      {},
    );

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between">
          <div>
            <CardTitle>
              {singleVariantMode
                ? "Performance Over Time"
                : "Variant Performance Over Time"}
            </CardTitle>
            <CardDescription>
              {singleVariantMode ? (
                <span>
                  Showing average metric values for <code>{metric_name}</code>
                </span>
              ) : (
                <span>
                  Showing average metric values by variant for metric{" "}
                  <code>{metric_name}</code>
                </span>
              )}
            </CardDescription>
          </div>
          <TimeGranularitySelector
            time_granularity={time_granularity}
            onTimeGranularityChange={onTimeGranularityChange}
          />
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-80 w-full">
            <BarChart accessibilityLayer data={data}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="date"
                tickLine={false}
                tickMargin={10}
                axisLine={true}
                tickFormatter={(value) =>
                  formatXAxisTimestamp(new Date(value), time_granularity)
                }
              />
              <YAxis
                tickLine={false}
                tickMargin={10}
                axisLine={true}
                tickFormatter={formatChartNumber}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(label) =>
                      formatTooltipTimestamp(new Date(label), time_granularity)
                    }
                    formatter={(value, name, entry) => {
                      const numInferences =
                        entry.payload[`${name}_num_inferences`];
                      return (
                        <div className="flex flex-1 items-center justify-between leading-none">
                          <span className="text-muted-foreground font-mono text-xs">
                            {name}
                          </span>
                          <div className="ml-2 grid text-right">
                            <span className="text-foreground font-mono font-medium tabular-nums">
                              {formatDetailedNumber(value as number)}
                            </span>
                            <span className="text-muted-foreground text-[10px]">
                              n={formatDetailedNumber(numInferences)}
                            </span>
                          </div>
                        </div>
                      );
                    }}
                  />
                }
              />
              <ChartLegend
                content={<ChartLegendContent className="font-mono text-xs" />}
              />
              {singleVariantMode ? (
                <Bar
                  key={variantNames[0]}
                  dataKey={variantNames[0]}
                  name={variantNames[0]}
                  fill={CHART_COLORS[0]}
                  radius={4}
                  maxBarSize={100}
                >
                  <ErrorBar
                    dataKey={`${variantNames[0]}_ci_error`}
                    strokeWidth={1}
                  />
                </Bar>
              ) : (
                variantNames.map((variantName) => (
                  <Bar
                    key={variantName}
                    dataKey={variantName}
                    name={variantName}
                    fill={chartConfig[variantName].color}
                    radius={4}
                    maxBarSize={100}
                  >
                    <ErrorBar
                      dataKey={`${variantName}_ci_error`}
                      strokeWidth={1}
                    />
                  </Bar>
                ))
              )}
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  );
}

// After you've already parsed `parsedRows`, you can group them by period_start
// and transform to the desired structure. For example:

export type VariantPerformanceData = {
  date: string;
  [key: string]: string | number; // Allow date as string and all other fields as numbers
};

type PerformanceDataGroupedByDate = {
  date: string;
  variants: Record<
    string,
    {
      num_inferences: number;
      avg_metric: number;
      stdev: number | null;
      ci_error: number | null;
    }
  >;
}[];

export function transformVariantPerformances(
  parsedRows: VariantPerformanceRow[],
): {
  data: VariantPerformanceData[];
  variantNames: string[];
} {
  // Remove rows with n=0 inferences
  const filtered = parsedRows.filter((row) => row.count > 0);

  const variantNames = [...new Set(filtered.map((row) => row.variant_name))];

  // First group by date
  const groupedByDate = filtered.reduce<PerformanceDataGroupedByDate>(
    (acc, row) => {
      const { period_start, variant_name, count, avg_metric, stdev, ci_error } =
        row;

      // See if we already have an entry for this period_start
      let existingEntry = acc.find((entry) => entry.date === period_start);
      if (!existingEntry) {
        existingEntry = {
          date: period_start,
          variants: {},
        };
        acc.push(existingEntry);
      }

      // Attach variant data under the variants key
      existingEntry.variants[variant_name] = {
        num_inferences: count,
        avg_metric,
        stdev,
        ci_error,
      };

      return acc;
    },
    [],
  );

  // Sort by date in descending order and take only the 10 most recent periods
  const sortedAndLimited = groupedByDate
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    .slice(0, 10)
    .reverse(); // Reverse back to chronological order for display

  // Convert to Recharts-friendly shape
  const data = sortedAndLimited.map((entry) => {
    const row: VariantPerformanceData = { date: entry.date };
    variantNames.forEach((variant) => {
      const vData = entry.variants[variant];
      row[variant] = vData?.avg_metric ?? 0;
      row[`${variant}_ci_error`] = vData?.ci_error ?? 0;
      row[`${variant}_num_inferences`] = vData?.num_inferences ?? 0;
    });
    return row;
  });

  return {
    data,
    variantNames,
  };
}
