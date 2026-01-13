import type { ModelUsageTimePoint, TimeWindow } from "~/types/tensorzero";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  formatChartNumber,
  formatDetailedNumber,
  formatXAxisTimestamp,
  formatTooltipTimestamp,
  CHART_COLORS,
  CHART_MARGIN,
  CHART_AXIS_STROKE,
} from "~/utils/chart";

import {
  BasicChartLegend,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "~/components/ui/chart";
import { TimeWindowSelector } from "~/components/ui/TimeWindowSelector";
import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";

export type ModelUsageMetric =
  | "inferences"
  | "input_tokens"
  | "output_tokens"
  | "total_tokens";

export const USAGE_METRIC_CONFIG = {
  inferences: {
    label: "Inferences",
    description: "Inference requests",
    formatter: (value: number) => `${formatDetailedNumber(value)} requests`,
  },
  input_tokens: {
    label: "Input Tokens",
    description: "Input tokens",
    formatter: (value: number) => `${formatDetailedNumber(value)} tokens`,
  },
  output_tokens: {
    label: "Output Tokens",
    description: "Output tokens",
    formatter: (value: number) => `${formatDetailedNumber(value)} tokens`,
  },
  total_tokens: {
    label: "Total Tokens",
    description: "Total tokens",
    formatter: (value: number) => `${formatDetailedNumber(value)} tokens`,
  },
} as const;

export function UsageTimeWindowSelector({
  value,
  onValueChange,
}: {
  value: TimeWindow;
  onValueChange: (value: TimeWindow) => void;
}) {
  return <TimeWindowSelector value={value} onValueChange={onValueChange} />;
}

export function UsageMetricSelector({
  value,
  onValueChange,
}: {
  value: ModelUsageMetric;
  onValueChange: (metric: ModelUsageMetric) => void;
}) {
  return (
    <Select
      value={value}
      onValueChange={(v: ModelUsageMetric) => onValueChange(v)}
    >
      <SelectTrigger>
        <SelectValue placeholder="Choose metric" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="inferences">Inferences</SelectItem>
        <SelectItem value="input_tokens">Input Tokens</SelectItem>
        <SelectItem value="output_tokens">Output Tokens</SelectItem>
        <SelectItem value="total_tokens">Total Tokens</SelectItem>
      </SelectContent>
    </Select>
  );
}

interface ModelUsageChartProps {
  modelUsageData: ModelUsageTimePoint[];
  selectedMetric: ModelUsageMetric;
  timeGranularity: TimeWindow;
}

export function ModelUsageChart({
  modelUsageData,
  selectedMetric,
  timeGranularity,
}: ModelUsageChartProps) {
  const { data, modelNames } = transformModelUsageData(
    modelUsageData,
    selectedMetric,
  );
  const chartConfig: Record<string, { label: string; color: string }> =
    modelNames.reduce(
      (config, modelName, index) => ({
        ...config,
        [modelName]: {
          label: modelName,
          color: CHART_COLORS[index % CHART_COLORS.length],
        },
      }),
      {},
    );

  return (
    <div>
      <ChartContainer config={chartConfig} className="h-72 w-full">
        <BarChart accessibilityLayer data={data} margin={CHART_MARGIN}>
          <CartesianGrid vertical={false} />
          {timeGranularity !== "cumulative" && (
            <XAxis
              dataKey="date"
              tickLine={false}
              tickMargin={10}
              axisLine={{ stroke: CHART_AXIS_STROKE }}
              tickFormatter={(value) =>
                formatXAxisTimestamp(new Date(value), timeGranularity)
              }
            />
          )}
          <YAxis
            tickLine={false}
            axisLine={false}
            tickFormatter={formatChartNumber}
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                labelFormatter={(label) =>
                  timeGranularity === "cumulative"
                    ? "Total"
                    : formatTooltipTimestamp(new Date(label), timeGranularity)
                }
                formatter={(value, name, entry) => {
                  const count = entry.payload[`${name}_count`];
                  const inputTokens = entry.payload[`${name}_input_tokens`];
                  const outputTokens = entry.payload[`${name}_output_tokens`];
                  const totalTokens = inputTokens + outputTokens;

                  return (
                    <div className="flex flex-1 items-center justify-between leading-none">
                      <span className="text-muted-foreground mr-2 font-mono text-xs">
                        {name}
                      </span>
                      <div className="grid text-right">
                        <span className="text-foreground font-mono font-medium tabular-nums">
                          {USAGE_METRIC_CONFIG[selectedMetric].formatter(
                            value as number,
                          )}
                        </span>
                        {selectedMetric === "inferences" && (
                          <span className="text-muted-foreground text-[10px]">
                            {formatDetailedNumber(totalTokens)} tokens
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
          {modelNames.map((modelName, index) => (
            <Bar
              key={modelName}
              dataKey={modelName}
              name={modelName}
              fill={CHART_COLORS[index % CHART_COLORS.length]}
              radius={4}
              maxBarSize={100}
            />
          ))}
        </BarChart>
      </ChartContainer>
      <BasicChartLegend items={modelNames} colors={CHART_COLORS} />
    </div>
  );
}

export type ModelUsageData = {
  date: string;
  [key: string]: string | number;
};

type UsageDataGroupedByDate = {
  date: string;
  models: Record<
    string,
    {
      count: number;
      input_tokens: number;
      output_tokens: number;
    }
  >;
}[];

export function transformModelUsageData(
  modelUsageData: ModelUsageTimePoint[],
  selectedMetric: ModelUsageMetric,
): {
  data: ModelUsageData[];
  modelNames: string[];
} {
  // Remove rows with count=0 or null
  const filtered = modelUsageData.filter(
    (row) => row.count && Number(row.count) > 0,
  );

  const modelNames = [...new Set(filtered.map((row) => row.model_name))];

  // First group by date
  const groupedByDate = filtered.reduce<UsageDataGroupedByDate>((acc, row) => {
    const { period_start, model_name, count, input_tokens, output_tokens } =
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
        models: {},
      };
      acc.push(existingEntry);
    }

    // Attach model data under the models key
    existingEntry.models[model_name] = {
      count: countNum,
      input_tokens: inputTokensNum,
      output_tokens: outputTokensNum,
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
    const row: ModelUsageData = { date: entry.date };
    modelNames.forEach((modelName) => {
      const modelData = entry.models[modelName];
      // Set the main value based on selected metric
      if (selectedMetric === "inferences") {
        row[modelName] = modelData?.count ?? 0;
      } else if (selectedMetric === "input_tokens") {
        row[modelName] = modelData?.input_tokens ?? 0;
      } else if (selectedMetric === "output_tokens") {
        row[modelName] = modelData?.output_tokens ?? 0;
      } else if (selectedMetric === "total_tokens") {
        row[modelName] =
          (modelData?.input_tokens ?? 0) + (modelData?.output_tokens ?? 0);
      }
      // Keep all data for tooltip
      row[`${modelName}_count`] = modelData?.count ?? 0;
      row[`${modelName}_input_tokens`] = modelData?.input_tokens ?? 0;
      row[`${modelName}_output_tokens`] = modelData?.output_tokens ?? 0;
    });
    return row;
  });

  return {
    data,
    modelNames,
  };
}
