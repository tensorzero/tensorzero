import type { ModelLatencyDatapoint, TimeWindow } from "tensorzero-node";
import {
  Line,
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import React, { useState, useMemo } from "react";
import { Await } from "react-router";
import { CHART_COLORS } from "~/utils/chart";
import {
  Select,
  SelectItem,
  SelectContent,
  SelectValue,
  SelectTrigger,
} from "~/components/ui/select";
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
} from "~/components/ui/chart";

type LatencyMetric = "response_time_ms" | "ttft_ms";

interface TooltipPayload {
  value: number | null;
  color: string;
  dataKey: string;
}

interface TooltipProps {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: number;
}

function CustomTooltipContent({ active, payload, label }: TooltipProps) {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="border-border/50 bg-background min-w-[10rem] rounded-lg border px-2.5 py-1.5 text-xs shadow-xl">
      <div className="flex items-center justify-between gap-2 pb-1">
        <span>Percentile</span>
        <span className="text-foreground font-mono font-medium tabular-nums">
          {((label || 0) * 100).toFixed(1)}%
        </span>
      </div>
      <div className="border-border/50 border-t pt-1.5">
        {payload.map((entry, index) => {
          if (!entry.value || entry.value <= 0) return null;
          return (
            <div
              key={index}
              className="flex items-center justify-between gap-4"
            >
              <div className="flex items-center gap-1.5">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-[2px]"
                  style={{
                    background: entry.color,
                    border: `1px solid ${entry.color}`,
                  }}
                />
                <span className="text-muted-foreground mr-2 font-mono text-xs">
                  {entry.dataKey}
                </span>
              </div>
              <span className="text-foreground font-mono font-medium tabular-nums">
                {Math.round(entry.value)}ms
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const MARGIN = { top: 12, right: 16, bottom: 28, left: 56 };

function LatencyTimeWindowSelector({
  value,
  onValueChange,
}: {
  value: TimeWindow;
  onValueChange: (timeWindow: TimeWindow) => void;
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger>
        <SelectValue placeholder="Choose time window" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="hour">Last Hour</SelectItem>
        <SelectItem value="day">Last Day</SelectItem>
        <SelectItem value="week">Last Week</SelectItem>
        <SelectItem value="month">Last Month</SelectItem>
        <SelectItem value="cumulative">All Time</SelectItem>
      </SelectContent>
    </Select>
  );
}

export function LatencyQuantileChart({
  latencyData,
  selectedMetric,
  quantiles,
}: {
  latencyData: ModelLatencyDatapoint[];
  selectedMetric: LatencyMetric;
  quantiles: number[];
}) {
  // Prepare eCDF series (your existing transform is fine)
  const { data, modelNames } = useMemo(
    () => transformLatencyData(latencyData, selectedMetric, quantiles),
    [latencyData, selectedMetric, quantiles],
  );

  const chartConfig: Record<string, { label: string; color: string }> = useMemo(
    () =>
      modelNames.reduce(
        (config, modelName, index) => ({
          ...config,
          [modelName]: {
            label: modelName,
            color: CHART_COLORS[index % CHART_COLORS.length],
          },
        }),
        {},
      ),
    [modelNames],
  );

  return (
    <ChartContainer config={chartConfig} className="h-80 w-full">
      <LineChart accessibilityLayer data={data} margin={MARGIN}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="percentile"
          domain={[0, 1]}
          tickLine={false}
          tickMargin={10}
          axisLine={true}
          ticks={[
            0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
          ]}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
        />
        <YAxis
          scale="log"
          domain={["dataMin", "dataMax"]}
          tickLine={false}
          tickMargin={10}
          axisLine={true}
          tickFormatter={(v) => `${v}ms`}
        />

        <Tooltip
          content={<CustomTooltipContent />}
          cursor={{
            stroke: "#666666",
            strokeDasharray: "3 3",
            strokeWidth: 2,
          }}
        />

        <ChartLegend
          content={<ChartLegendContent className="font-mono text-xs" />}
        />

        {modelNames.map((name, index) => (
          <Line
            key={name}
            type="monotone"
            dataKey={name}
            name={name}
            stroke={CHART_COLORS[index % CHART_COLORS.length]}
            strokeWidth={2}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ChartContainer>
  );
}

type QuantileDataPoint = {
  percentile: number;
  [modelName: string]: number | null;
};

function transformLatencyData(
  latencyData: ModelLatencyDatapoint[],
  selectedMetric: LatencyMetric,
  quantiles: number[],
): { data: QuantileDataPoint[]; modelNames: string[] } {
  const modelNames = latencyData.map((d) => d.model_name);
  const data: QuantileDataPoint[] = [];

  // Create data points for each quantile/percentile
  quantiles.forEach((percentile) => {
    const dp: QuantileDataPoint = { percentile };

    modelNames.forEach((name) => {
      const md = latencyData.find((d) => d.model_name === name)!;
      const arr =
        selectedMetric === "response_time_ms"
          ? md.response_time_ms_quantiles
          : md.ttft_ms_quantiles;

      // Find the quantile index for this percentile
      const quantileIndex = quantiles.indexOf(percentile);
      if (quantileIndex >= 0 && quantileIndex < arr.length) {
        const latencyValue = arr[quantileIndex];
        dp[name] = latencyValue && latencyValue > 0 ? latencyValue : null;
      } else {
        dp[name] = null;
      }
    });
    data.push(dp);
  });

  return { data, modelNames };
}

export function ModelLatency({
  modelLatencyDataPromise,
  quantiles,
  timeGranularity,
  onTimeGranularityChange,
}: {
  modelLatencyDataPromise: Promise<ModelLatencyDatapoint[]>;
  quantiles: number[];
  timeGranularity: TimeWindow;
  onTimeGranularityChange: (granularity: TimeWindow) => void;
}) {
  const [selectedMetric, setSelectedMetric] =
    useState<LatencyMetric>("response_time_ms");

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between">
        <div>
          <CardTitle>Model Latency Distribution</CardTitle>
          <CardDescription>
            Quantiles of latency metrics by model
          </CardDescription>
        </div>
        <div className="flex flex-col justify-center gap-2">
          <LatencyTimeWindowSelector
            value={timeGranularity}
            onValueChange={onTimeGranularityChange}
          />
          <Select
            value={selectedMetric}
            onValueChange={(value: LatencyMetric) => setSelectedMetric(value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Choose metric" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="response_time_ms">Response Time</SelectItem>
              <SelectItem value="ttft_ms">Time to First Token</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        <React.Suspense fallback={<div>Loading latency data...</div>}>
          <Await resolve={modelLatencyDataPromise}>
            {(latencyData) => (
              <LatencyQuantileChart
                latencyData={latencyData}
                selectedMetric={selectedMetric}
                quantiles={quantiles}
              />
            )}
          </Await>
        </React.Suspense>
      </CardContent>
    </Card>
  );
}
