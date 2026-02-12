import type { VariantCost } from "~/types/tensorzero";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  CHART_COLORS,
  costDecimalPlaces,
  formatCostForChart,
  formatXAxisTimestamp,
  formatTooltipTimestamp,
} from "~/utils/chart";

import { Card, CardContent, CardHeader } from "~/components/ui/card";
import {
  ChartContainer,
  ChartLegendList,
  ChartTooltip,
} from "~/components/ui/chart";
import { TimeGranularitySelector } from "./TimeGranularitySelector";
import { useTimeGranularityParam } from "~/hooks/use-time-granularity-param";

const styles = {
  empty: "text-muted-foreground py-8 text-center text-sm",
  emptySummary: "text-muted-foreground mt-2 text-center text-xs",
  coverage: "text-muted-foreground mt-3 text-center text-xs",
  totalCost: "text-muted-foreground mb-2 text-center text-sm font-medium",
};

export type VariantCostData = {
  date: string;
  [key: string]: string | number;
};

export function transformVariantCost(parsedRows: VariantCost[]): {
  data: VariantCostData[];
  variantNames: string[];
} {
  const variantNames = [
    ...new Set(parsedRows.map((row) => row.variant_name)),
  ].sort();

  const groupedByDate = parsedRows.reduce<
    Record<string, Record<string, number>>
  >((acc, row) => {
    const { period_start, variant_name, total_cost } = row;

    if (!acc[period_start]) {
      acc[period_start] = {};
    }

    acc[period_start][variant_name] = total_cost;
    return acc;
  }, {});

  const data = Object.entries(groupedByDate)
    .map(([date, variants]) => {
      const row: VariantCostData = { date };
      variantNames.forEach((variant) => {
        row[variant] = variants[variant] ?? 0;
      });
      return row;
    })
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    .slice(-10);

  return {
    data,
    variantNames,
  };
}

export function VariantCostChart({
  variant_cost,
  totalInferenceCount,
}: {
  variant_cost: VariantCost[];
  /** When cost array is empty, use this to show fraction of inferences without cost (e.g. function page). */
  totalInferenceCount?: number;
}) {
  const [time_granularity, onTimeGranularityChange] = useTimeGranularityParam(
    "cost_time_granularity",
    "week",
  );

  if (variant_cost.length === 0) {
    return (
      <Card>
        <CardContent>
          <p className={styles.empty}>
            No cost data yet. Cost is recorded when models have cost
            configuration.
          </p>
          {totalInferenceCount !== undefined && totalInferenceCount > 0 && (
            <p className={styles.emptySummary}>
              Total cost: $0. 0 of {totalInferenceCount.toLocaleString()}{" "}
              inferences have cost data (100% without cost).
            </p>
          )}
        </CardContent>
      </Card>
    );
  }

  const { data, variantNames } = transformVariantCost(variant_cost);

  const totalInferences = variant_cost.reduce(
    (sum, row) => sum + (row.inference_count ?? 0),
    0,
  );
  const inferencesWithCost = variant_cost.reduce(
    (sum, row) => sum + (row.inferences_with_cost ?? 0),
    0,
  );

  const totalCostByVariant = variant_cost.reduce<Record<string, number>>(
    (acc, row) => {
      const name = row.variant_name;
      acc[name] = (acc[name] ?? 0) + (row.total_cost ?? 0);
      return acc;
    },
    {},
  );

  const maxCostInLegend = Math.max(0, ...Object.values(totalCostByVariant));
  const maxCostInChart = Math.max(
    0,
    ...data.flatMap((row) =>
      variantNames.map((name) => Number(row[name]) || 0),
    ),
  );
  const maxCost = Math.max(maxCostInLegend, maxCostInChart);
  const costDecimals = costDecimalPlaces(maxCost);

  const totalCost = Object.values(totalCostByVariant).reduce(
    (sum, v) => sum + v,
    0,
  );

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
          <div className="flex flex-col gap-1">
            <p className={styles.totalCost}>
              Total cost: {formatCostForChart(totalCost, costDecimals)}
            </p>
            <TimeGranularitySelector
              time_granularity={time_granularity}
              onTimeGranularityChange={onTimeGranularityChange}
              includeCumulative={true}
            />
          </div>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig}>
            <AreaChart accessibilityLayer data={data}>
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
                width={88}
                tickLine={false}
                tickMargin={10}
                axisLine={true}
                label={{
                  value: "Cost ($)",
                  angle: -90,
                  position: "outsideLeft",
                }}
                tickFormatter={(value) =>
                  formatCostForChart(Number(value), costDecimals)
                }
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
                        {formatTooltipTimestamp(
                          new Date(label),
                          time_granularity,
                        )}
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
                                  {formatCostForChart(Number(entry.value), costDecimals)}
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
                              {formatCostForChart(total, costDecimals)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                }}
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
          <ChartLegendList
            items={variantNames}
            colors={CHART_COLORS}
            valueByKey={totalCostByVariant}
            formatValue={(n) => formatCostForChart(n, costDecimals)}
          />
          {totalInferences > 0 && (
            <p className={styles.coverage}>
              {inferencesWithCost === totalInferences ? (
                <>All {totalInferences.toLocaleString()} inferences have cost data.</>
              ) : (
                <>
                  {inferencesWithCost.toLocaleString()} of{" "}
                  {totalInferences.toLocaleString()} inferences have cost data (
                  {Math.round(
                    (100 * (totalInferences - inferencesWithCost)) /
                      totalInferences,
                  )}
                  % without cost).
                </>
              )}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
