import type { VariantPerformanceRow, TimeWindow } from "~/types/tensorzero";
import { Bar, BarChart, ErrorBar, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  formatChartNumber,
  formatDetailedNumber,
  formatXAxisTimestamp,
  formatTooltipTimestamp,
  getChartColor,
} from "~/utils/chart";
import {
  ChartContainer,
  ChartLegend,
  ChartTooltip,
  ChartTooltipContent,
} from "~/components/ui/chart";

export type VariantPerformanceData = {
  date: string;
  [key: string]: string | number;
};

interface VariantPerformanceChartProps {
  data: VariantPerformanceData[];
  variantNames: string[];
  timeGranularity: TimeWindow;
  singleVariantMode: boolean;
}

export function VariantPerformanceChart({
  data,
  variantNames,
  timeGranularity,
  singleVariantMode,
}: VariantPerformanceChartProps) {
  const chartConfig: Record<string, { label: string; color: string }> =
    variantNames.reduce(
      (config, variantName, index) => ({
        ...config,
        [variantName]: {
          label: variantName,
          color: singleVariantMode ? getChartColor(0) : getChartColor(index),
        },
      }),
      {},
    );

  return (
    <>
      <ChartContainer config={chartConfig}>
        <BarChart accessibilityLayer data={data}>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="date"
            tickLine={false}
            tickMargin={10}
            axisLine={true}
            tickFormatter={(value) =>
              formatXAxisTimestamp(new Date(value), timeGranularity)
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
                  formatTooltipTimestamp(new Date(label), timeGranularity)
                }
                formatter={(value, name, entry) => {
                  const numInferences = entry.payload[`${name}_num_inferences`];
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
          {singleVariantMode ? (
            <Bar
              key={variantNames[0]}
              dataKey={variantNames[0]}
              name={variantNames[0]}
              fill={getChartColor(0)}
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
                <ErrorBar dataKey={`${variantName}_ci_error`} strokeWidth={1} />
              </Bar>
            ))
          )}
        </BarChart>
      </ChartContainer>
      <ChartLegend
        items={variantNames}
        colors={
          singleVariantMode
            ? [getChartColor(0)]
            : variantNames.map(
                (name) => chartConfig[name]?.color ?? getChartColor(0),
              )
        }
      />
    </>
  );
}

type PerformanceDataGroupedByDate = {
  date: string;
  variants: Record<
    string,
    {
      num_inferences: number;
      avg_metric: number;
      stdev?: number;
      ci_error?: number;
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

  const variantNames = [
    ...new Set(filtered.map((row) => row.variant_name)),
  ].sort();

  // First group by date
  const groupedByDate = filtered.reduce<PerformanceDataGroupedByDate>(
    (acc, row) => {
      const { period_start, variant_name, count, avg_metric, stdev, ci_error } =
        row;

      let existingEntry = acc.find((entry) => entry.date === period_start);
      if (!existingEntry) {
        existingEntry = {
          date: period_start,
          variants: {},
        };
        acc.push(existingEntry);
      }

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
