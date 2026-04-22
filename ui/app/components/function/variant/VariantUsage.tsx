import type { VariantUsageTimePoint } from "~/types/tensorzero";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  formatChartNumber,
  formatDetailedNumber,
  formatXAxisTimestamp,
  formatTooltipTimestamp,
  CHART_COLORS,
} from "~/utils/chart";
import { useState, Suspense } from "react";
import { Await, useAsyncError, isRouteErrorResponse } from "react-router";
import { SectionErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";
import { AlertCircle, AlertTriangle } from "lucide-react";
import { formatCost } from "~/utils/cost";

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
  ChartTooltip,
  ChartTooltipContent,
} from "~/components/ui/chart";
import { TimeGranularitySelector } from "./TimeGranularitySelector";
import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";
import { useTimeGranularityParam } from "~/hooks/use-time-granularity-param";

export type VariantUsageMetric =
  | "inferences"
  | "input_tokens"
  | "output_tokens"
  | "total_tokens"
  | "cost";

const METRIC_TYPE_CONFIG = {
  inferences: {
    label: "Inferences",
    description: "Number of inference requests",
    formatter: (value: number) => `${formatDetailedNumber(value)} requests`,
  },
  input_tokens: {
    label: "Input Tokens",
    description: "Input token usage",
    formatter: (value: number) => `${formatDetailedNumber(value)} tokens`,
  },
  output_tokens: {
    label: "Output Tokens",
    description: "Output token usage",
    formatter: (value: number) => `${formatDetailedNumber(value)} tokens`,
  },
  total_tokens: {
    label: "Total Tokens",
    description: "Total token usage (input + output)",
    formatter: (value: number) => `${formatDetailedNumber(value)} tokens`,
  },
  cost: {
    label: "Cost",
    description: "Estimated cost",
    formatter: (value: number) => formatCost(value),
  },
} as const;

function VariantUsageError() {
  const error = useAsyncError();
  let message = "Failed to load variant usage data";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading variant usage"
      description={message}
    />
  );
}

function MetricSelector({
  selectedMetric,
  onMetricChange,
}: {
  selectedMetric: VariantUsageMetric;
  onMetricChange: (metric: VariantUsageMetric) => void;
}) {
  return (
    <Select
      value={selectedMetric}
      onValueChange={(value: VariantUsageMetric) => onMetricChange(value)}
    >
      <SelectTrigger>
        <SelectValue placeholder="Choose metric" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="inferences">Inferences</SelectItem>
        <SelectItem value="input_tokens">Input Tokens</SelectItem>
        <SelectItem value="output_tokens">Output Tokens</SelectItem>
        <SelectItem value="total_tokens">Total Tokens</SelectItem>
        <SelectItem value="cost">Cost</SelectItem>
      </SelectContent>
    </Select>
  );
}

export function VariantUsage({
  variantUsageDataPromise,
}: {
  variantUsageDataPromise: Promise<VariantUsageTimePoint[]>;
}) {
  const [timeGranularity, onTimeGranularityChange] = useTimeGranularityParam(
    "variantUsageTimeGranularity",
    "week",
  );
  const [selectedMetric, setSelectedMetric] =
    useState<VariantUsageMetric>("inferences");

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between">
        <div>
          <CardTitle>Variant Usage Over Time</CardTitle>
          <CardDescription>
            {METRIC_TYPE_CONFIG[selectedMetric].description} by variant
          </CardDescription>
        </div>
        <div className="flex items-center gap-2">
          <TimeGranularitySelector
            time_granularity={timeGranularity}
            onTimeGranularityChange={onTimeGranularityChange}
          />
          <MetricSelector
            selectedMetric={selectedMetric}
            onMetricChange={setSelectedMetric}
          />
        </div>
      </CardHeader>
      <CardContent>
        <Suspense fallback={<div>Loading variant usage data...</div>}>
          <Await
            resolve={variantUsageDataPromise}
            errorElement={<VariantUsageError />}
          >
            {(variantUsageData) => {
              const { data, variantNames, visiblePeriods } =
                transformVariantUsageData(variantUsageData, selectedMetric);
              const chartConfig: Record<
                string,
                { label: string; color: string }
              > = variantNames.reduce(
                (config, variantName, index) => ({
                  ...config,
                  [variantName]: {
                    label: variantName,
                    color: CHART_COLORS[index % CHART_COLORS.length],
                  },
                }),
                {},
              );

              // Compute cost coverage percentage using backend-provided count_with_cost,
              // limited to the visible periods shown in the chart
              let costCoveragePercent: number | null = null;
              if (selectedMetric === "cost") {
                const visibleRows = variantUsageData.filter(
                  (row) =>
                    row.count &&
                    Number(row.count) > 0 &&
                    visiblePeriods.has(row.period_start),
                );
                let totalCount = 0;
                let countWithCost = 0;
                for (const row of visibleRows) {
                  totalCount += Number(row.count);
                  countWithCost += Number(row.count_with_cost ?? 0);
                }
                if (totalCount > 0) {
                  costCoveragePercent = Math.floor(
                    (countWithCost / totalCount) * 100,
                  );
                }
              }

              return (
                <>
                  {selectedMetric === "cost" &&
                    costCoveragePercent != null &&
                    costCoveragePercent < 100 && (
                      <div className="mb-4 flex items-center gap-2 rounded-md border border-yellow-500 bg-yellow-50 p-3 text-sm text-yellow-700 dark:bg-yellow-950 dark:text-yellow-200">
                        <AlertTriangle className="h-4 w-4 shrink-0 text-yellow-500" />
                        <span>
                          Cost data only covers ~{costCoveragePercent}% of
                          variant inferences. Some variants may not have cost
                          tracking configured.
                        </span>
                      </div>
                    )}
                  <ChartContainer config={chartConfig}>
                    <BarChart accessibilityLayer data={data}>
                      <CartesianGrid vertical={false} />
                      {timeGranularity !== "cumulative" && (
                        <XAxis
                          dataKey="date"
                          tickLine={false}
                          tickMargin={10}
                          axisLine={true}
                          tickFormatter={(value) =>
                            formatXAxisTimestamp(
                              new Date(value),
                              timeGranularity,
                            )
                          }
                        />
                      )}
                      <YAxis
                        tickLine={false}
                        tickMargin={10}
                        axisLine={true}
                        width={selectedMetric === "cost" ? 90 : 60}
                        tickFormatter={(value) =>
                          selectedMetric === "cost"
                            ? formatCost(value)
                            : formatChartNumber(value)
                        }
                      />
                      <ChartTooltip
                        content={
                          <ChartTooltipContent
                            labelFormatter={(label) =>
                              timeGranularity === "cumulative"
                                ? "Total"
                                : formatTooltipTimestamp(
                                    new Date(label),
                                    timeGranularity,
                                  )
                            }
                            formatter={(value, name, entry) => {
                              const count = entry.payload[`${name}_count`];
                              const inputTokens =
                                entry.payload[`${name}_input_tokens`];
                              const outputTokens =
                                entry.payload[`${name}_output_tokens`];
                              const totalTokens = inputTokens + outputTokens;

                              return (
                                <div className="flex flex-1 items-center justify-between leading-none">
                                  <span className="text-muted-foreground mr-2 font-mono text-xs">
                                    {name}
                                  </span>
                                  <div className="grid text-right">
                                    <span className="text-foreground font-mono font-medium tabular-nums">
                                      {METRIC_TYPE_CONFIG[
                                        selectedMetric
                                      ].formatter(value as number)}
                                    </span>
                                    {selectedMetric === "inferences" && (
                                      <span className="text-muted-foreground text-[10px]">
                                        {formatDetailedNumber(totalTokens)}{" "}
                                        tokens
                                      </span>
                                    )}
                                    {selectedMetric !== "inferences" && (
                                      <span className="text-muted-foreground text-[10px]">
                                        {formatDetailedNumber(count)} requests
                                      </span>
                                    )}
                                  </div>
                                </div>
                              );
                            }}
                          />
                        }
                      />
                      {variantNames.map((variantName, index) => (
                        <Bar
                          key={variantName}
                          dataKey={variantName}
                          name={variantName}
                          fill={CHART_COLORS[index % CHART_COLORS.length]}
                          radius={4}
                          maxBarSize={100}
                        />
                      ))}
                    </BarChart>
                  </ChartContainer>
                  <ChartLegend items={variantNames} colors={CHART_COLORS} />
                </>
              );
            }}
          </Await>
        </Suspense>
      </CardContent>
    </Card>
  );
}

export type VariantUsageData = {
  date: string;
  [key: string]: string | number;
};

type UsageDataGroupedByDate = {
  date: string;
  variants: Record<
    string,
    {
      count: number;
      input_tokens: number;
      output_tokens: number;
      cost: number | null;
    }
  >;
}[];

export function transformVariantUsageData(
  variantUsageData: VariantUsageTimePoint[],
  selectedMetric: VariantUsageMetric,
): {
  data: VariantUsageData[];
  variantNames: string[];
  visiblePeriods: Set<string>;
} {
  // Remove rows with count=0 or null
  const filtered = variantUsageData.filter(
    (row) => row.count && Number(row.count) > 0,
  );

  const variantNames = [...new Set(filtered.map((row) => row.variant_name))];

  // First group by date
  const groupedByDate = filtered.reduce<UsageDataGroupedByDate>((acc, row) => {
    const { period_start, variant_name, count, input_tokens, output_tokens } =
      row;

    // Convert bigints to numbers, handling null values
    const countNum = count ? Number(count) : 0;
    const inputTokensNum = input_tokens ? Number(input_tokens) : 0;
    const outputTokensNum = output_tokens ? Number(output_tokens) : 0;

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
      count: countNum,
      input_tokens: inputTokensNum,
      output_tokens: outputTokensNum,
      cost: row.cost,
    };

    return acc;
  }, []);

  // Sort by date in descending order and take only the 10 most recent periods
  const sortedAndLimited = groupedByDate
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    .slice(0, 10)
    .reverse(); // Reverse back to chronological order for display

  // Convert to Recharts-friendly shape
  const data = sortedAndLimited.map((entry) => {
    const row: VariantUsageData = { date: entry.date };
    variantNames.forEach((variantName) => {
      const variantData = entry.variants[variantName];
      // Set the main value based on selected metric
      if (selectedMetric === "inferences") {
        row[variantName] = variantData?.count ?? 0;
      } else if (selectedMetric === "input_tokens") {
        row[variantName] = variantData?.input_tokens ?? 0;
      } else if (selectedMetric === "output_tokens") {
        row[variantName] = variantData?.output_tokens ?? 0;
      } else if (selectedMetric === "total_tokens") {
        row[variantName] =
          (variantData?.input_tokens ?? 0) + (variantData?.output_tokens ?? 0);
      } else if (selectedMetric === "cost") {
        row[variantName] = variantData?.cost ?? 0;
      }
      // Keep all data for tooltip
      row[`${variantName}_count`] = variantData?.count ?? 0;
      row[`${variantName}_input_tokens`] = variantData?.input_tokens ?? 0;
      row[`${variantName}_output_tokens`] = variantData?.output_tokens ?? 0;
      row[`${variantName}_cost`] = variantData?.cost ?? 0;
    });
    return row;
  });

  // When cost is selected, filter to only variants with at least one non-null cost value
  // in the visible periods (sortedAndLimited, not the full groupedByDate)
  const filteredVariantNames =
    selectedMetric === "cost"
      ? variantNames.filter((variantName) =>
          sortedAndLimited.some(
            (entry) => entry.variants[variantName]?.cost != null,
          ),
        )
      : variantNames;

  const visiblePeriods = new Set(sortedAndLimited.map((entry) => entry.date));

  return {
    data,
    variantNames: filteredVariantNames,
    visiblePeriods,
  };
}
