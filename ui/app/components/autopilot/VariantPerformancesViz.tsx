import { Bar, BarChart, ErrorBar, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  formatChartNumber,
  formatDetailedNumber,
  formatXAxisTimestamp,
  formatTooltipTimestamp,
  CHART_COLORS,
} from "~/utils/chart";
import {
  ChartContainer,
  ChartLegend,
  ChartTooltip,
  ChartTooltipContent,
} from "~/components/ui/chart";
import type {
  VariantPerformancesVisualization,
  TimeWindow,
} from "~/types/tensorzero";
import { transformVariantPerformances } from "~/components/function/variant/VariantPerformance";

type VariantPerformancesVizProps = {
  data: VariantPerformancesVisualization;
};

export default function VariantPerformancesViz({
  data,
}: VariantPerformancesVizProps) {
  const VALID_TIME_WINDOWS: Set<string> = new Set([
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "cumulative",
  ]);
  const timeGranularity: TimeWindow = VALID_TIME_WINDOWS.has(
    data.time_granularity,
  )
    ? (data.time_granularity as TimeWindow)
    : "week";
  const { data: chartData, variantNames } = transformVariantPerformances(
    data.performances,
  );

  if (chartData.length === 0) {
    return (
      <div className="text-fg-muted py-4 text-center text-sm">
        No variant performance data available
      </div>
    );
  }

  const singleVariantMode = variantNames.length === 1;

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
    <div className="flex flex-col gap-1">
      <div className="text-fg-secondary text-xs font-medium">
        <code>{data.metric_name}</code> for <code>{data.function_name}</code>
      </div>
      <ChartContainer config={chartConfig}>
        <BarChart accessibilityLayer data={chartData}>
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
            ? [CHART_COLORS[0]]
            : variantNames.map(
                (name) => chartConfig[name]?.color ?? CHART_COLORS[0],
              )
        }
      />
    </div>
  );
}
