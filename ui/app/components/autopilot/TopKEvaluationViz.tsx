import { useMemo } from "react";
import {
  Bar,
  BarChart,
  ComposedChart,
  ErrorBar,
  Line,
  ReferenceLine,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { ChartContainer, ChartTooltip } from "~/components/ui/chart";
import type { TopKEvaluationVisualization } from "~/types/tensorzero";

type TopKEvaluationVizProps = {
  data: TopKEvaluationVisualization;
};

type ChartDataPoint = {
  name: string;
  mean: number;
  lower: number;
  upper: number;
  // Error bar uses [lowerError, upperError] for asymmetric errors
  error: [number, number];
  count: number;
};

function formatNumber(value: number): string {
  return value.toFixed(3);
}

function PerformanceTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: ChartDataPoint }>;
}) {
  if (!active || !payload?.length) {
    return null;
  }

  const data = payload[0].payload;

  return (
    <div className="border-border bg-bg-secondary rounded-md border p-2 text-xs shadow-lg">
      <div className="mb-1 font-medium">{data.name}</div>
      <div className="text-fg-secondary space-y-0.5">
        <div>
          Mean: <span className="font-mono">{formatNumber(data.mean)}</span>
        </div>
        <div>
          CI: [<span className="font-mono">{formatNumber(data.lower)}</span>,{" "}
          <span className="font-mono">{formatNumber(data.upper)}</span>]
        </div>
      </div>
    </div>
  );
}

function CountTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: ChartDataPoint }>;
}) {
  if (!active || !payload?.length) {
    return null;
  }

  const data = payload[0].payload;

  return (
    <div className="border-border bg-bg-secondary rounded-md border p-2 text-xs shadow-lg">
      <div className="mb-1 font-medium">{data.name}</div>
      <div className="text-fg-secondary">
        Samples:{" "}
        <span className="font-mono">{data.count.toLocaleString()}</span>
      </div>
    </div>
  );
}

export default function TopKEvaluationViz({ data }: TopKEvaluationVizProps) {
  const chartData = useMemo(() => {
    const entries = Object.entries(data.variant_summaries);

    // Sort by mean estimate (descending - best first)
    const sorted = entries.sort(([, a], [, b]) => {
      const aMean = a?.mean_est ?? 0;
      const bMean = b?.mean_est ?? 0;
      return bMean - aMean;
    });

    return sorted.map(([name, summary]): ChartDataPoint => {
      const mean = summary?.mean_est ?? 0;
      const lower = summary?.cs_lower ?? 0;
      const upper = summary?.cs_upper ?? 0;
      // Convert bigint to number for chart rendering
      const count = Number(summary?.count ?? 0);

      return {
        name,
        mean,
        lower,
        upper,
        // Error bar values are relative to the mean: [lower error, upper error]
        error: [mean - lower, upper - mean],
        count,
      };
    });
  }, [data.variant_summaries]);

  // Calculate max count for bar chart domain
  const maxCount = useMemo(() => {
    if (chartData.length === 0) return 100;
    const max = Math.max(...chartData.map((d) => d.count));
    return Math.ceil(max * 1.15); // Add 15% padding for labels
  }, [chartData]);

  if (chartData.length === 0) {
    return (
      <div className="text-fg-muted py-4 text-center text-sm">
        No variant data available
      </div>
    );
  }

  const chartMargin = { top: 10, right: 20, left: 10, bottom: 5 };
  const bottomChartMargin = { top: 10, right: 20, left: 10, bottom: 60 };

  return (
    <div className="flex flex-col gap-0">
      {/* Top chart: Mean performance with confidence intervals */}
      <div className="text-fg-secondary mb-1 text-xs font-medium">
        Mean Performance
      </div>
      <ChartContainer config={{}} className="h-[200px] w-full">
        <ComposedChart data={chartData} margin={chartMargin}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#d1d5db"
            strokeOpacity={0.8}
          />
          <XAxis
            dataKey="name"
            tick={false}
            axisLine={{
              stroke: "#d1d5db",
              strokeDasharray: "3 3",
              strokeOpacity: 0.8,
            }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={formatNumber}
            tick={{ fontSize: 10, fill: "#6b7280" }}
            axisLine={{
              stroke: "#d1d5db",
              strokeDasharray: "3 3",
              strokeOpacity: 0.8,
            }}
            tickLine={{ stroke: "#d1d5db", strokeOpacity: 0.8 }}
            width={50}
          />
          <ChartTooltip content={<PerformanceTooltip />} />
          <ReferenceLine
            y={0.5}
            stroke="var(--border)"
            strokeDasharray="3 3"
            opacity={0.5}
          />
          <Line
            type="linear"
            dataKey="mean"
            stroke="transparent"
            dot={{ fill: "#3b82f6", r: 5 }}
            activeDot={{ fill: "#3b82f6", r: 7 }}
            isAnimationActive={false}
          >
            <ErrorBar
              dataKey="error"
              direction="y"
              width={8}
              strokeWidth={2}
              stroke="#ec4899"
            />
          </Line>
        </ComposedChart>
      </ChartContainer>

      {/* Bottom chart: Number of evaluations */}
      <div className="text-fg-secondary mt-2 mb-1 text-xs font-medium">
        Number of Evaluations
      </div>
      <ChartContainer config={{}} className="h-[180px] w-full">
        <BarChart data={chartData} margin={bottomChartMargin}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#d1d5db"
            strokeOpacity={0.8}
            vertical={false}
          />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 10, fill: "#6b7280" }}
            axisLine={{
              stroke: "#d1d5db",
              strokeDasharray: "3 3",
              strokeOpacity: 0.8,
            }}
            tickLine={{ stroke: "#d1d5db", strokeOpacity: 0.8 }}
            angle={-45}
            textAnchor="end"
            height={60}
            interval={0}
          />
          <YAxis
            domain={[0, maxCount]}
            tick={{ fontSize: 10, fill: "#6b7280" }}
            axisLine={{
              stroke: "#d1d5db",
              strokeDasharray: "3 3",
              strokeOpacity: 0.8,
            }}
            tickLine={{ stroke: "#d1d5db", strokeOpacity: 0.8 }}
            width={50}
          />
          <ChartTooltip content={<CountTooltip />} />
          <Bar
            dataKey="count"
            fill="var(--green-500)"
            opacity={0.8}
            radius={[4, 4, 0, 0]}
            label={{
              position: "top",
              fontSize: 9,
              fill: "var(--fg-secondary)",
            }}
          />
        </BarChart>
      </ChartContainer>
    </div>
  );
}
