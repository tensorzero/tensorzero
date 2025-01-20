import type { VariantPerformanceRow } from "~/utils/clickhouse/function";
// import { TrendingUp } from "lucide-react";
import { Bar, BarChart, ErrorBar, CartesianGrid, XAxis } from "recharts";

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

const CHART_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
] as const;

export function VariantPerformance({
  variant_performances,
  metric_name,
}: {
  variant_performances: VariantPerformanceRow[];
  metric_name: string;
}) {
  const { data, variantNames } =
    transformVariantPerformances(variant_performances);

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
        <CardHeader>
          <CardTitle>Variant Performance Over Time</CardTitle>
          <CardDescription>
            Showing average metric values by variant for metric{" "}
            <code>{metric_name}</code>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="min-h-[200px] w-full">
            <BarChart accessibilityLayer data={data} height={200}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="date"
                tickLine={false}
                tickMargin={10}
                axisLine={false}
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent />} />
              {variantNames.map((variantName) => (
                <Bar
                  key={variantName}
                  dataKey={variantName}
                  name={variantName}
                  fill={chartConfig[variantName].color}
                  radius={4}
                >
                  <ErrorBar
                    dataKey={`${variantName}_ci_error`}
                    strokeWidth={1}
                  />
                </Bar>
              ))}
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

export function transformVariantPerformances(
  parsedRows: VariantPerformanceRow[],
): {
  data: VariantPerformanceData[];
  variantNames: string[];
} {
  const variantNames = [...new Set(parsedRows.map((row) => row.variant_name))];

  // First group by date
  const groupedByDate = parsedRows.reduce(
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
    [] as {
      date: string;
      variants: Record<
        string,
        {
          num_inferences: number;
          avg_metric: number;
          stdev: number;
          ci_error: number;
        }
      >;
    }[],
  );

  // Convert to Recharts-friendly shape
  const data = groupedByDate.map((entry) => {
    const row: VariantPerformanceData = { date: entry.date };
    variantNames.forEach((variant) => {
      const vData = entry.variants[variant];
      row[variant] = vData?.avg_metric ?? 0;
      row[`${variant}_ci_error`] = vData?.ci_error ?? 0;
    });
    return row;
  });
  console.log(data);

  return {
    data,
    variantNames,
  };
}
