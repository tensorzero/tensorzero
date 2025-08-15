import type { TimeWindow, ModelUsageTimePoint } from "tensorzero-node";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { formatChartNumber, formatDetailedNumber } from "~/utils/chart";

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
import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";

const CHART_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
] as const;

export function ModelUsage({
  modelUsageData,
  timeGranularity,
  onTimeGranularityChange,
}: {
  modelUsageData: ModelUsageTimePoint[];
  timeGranularity: TimeWindow;
  onTimeGranularityChange: (timeGranularity: TimeWindow) => void;
}) {
  const { data, modelNames } = transformModelUsageData(modelUsageData);

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
    <div className="space-y-8">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between">
          <div>
            <CardTitle>Model Usage Over Time</CardTitle>
            <CardDescription>
              Showing request counts and token usage by model
            </CardDescription>
          </div>
          <div className="flex flex-col justify-center">
            <Select
              value={timeGranularity}
              onValueChange={onTimeGranularityChange}
            >
              <SelectTrigger>
                <SelectValue placeholder="Choose time granularity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="hour">Hour</SelectItem>
                <SelectItem value="day">Day</SelectItem>
                <SelectItem value="week">Week</SelectItem>
                <SelectItem value="month">Month</SelectItem>
                <SelectItem value="cumulative">Cumulative</SelectItem>
              </SelectContent>
            </Select>
          </div>
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
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
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
                      new Date(label).toLocaleDateString()
                    }
                    formatter={(value, name, entry) => {
                      const inputTokens = entry.payload[`${name}_input_tokens`];
                      const outputTokens =
                        entry.payload[`${name}_output_tokens`];
                      const totalTokens = inputTokens + outputTokens;

                      return (
                        <div className="flex flex-1 items-center justify-between leading-none">
                          <span className="text-muted-foreground">{name}</span>
                          <div className="grid text-right">
                            <span className="text-foreground font-mono font-medium tabular-nums">
                              {formatDetailedNumber(value as number)} requests
                            </span>
                            <span className="text-muted-foreground text-[10px]">
                              {formatDetailedNumber(totalTokens)} tokens
                            </span>
                          </div>
                        </div>
                      );
                    }}
                  />
                }
              />
              <ChartLegend content={<ChartLegendContent />} />
              {modelNames.map((modelName) => (
                <Bar
                  key={modelName}
                  dataKey={modelName}
                  name={modelName}
                  fill={chartConfig[modelName].color}
                  radius={4}
                  maxBarSize={100}
                />
              ))}
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>
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
      row[modelName] = modelData?.count ?? 0;
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
