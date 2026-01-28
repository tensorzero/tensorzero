import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { ChartContainer, ChartTooltip } from "~/components/ui/chart";
import type { TopKEvaluationVisualization } from "~/types/tensorzero";
import { CHART_COLORS } from "~/utils/chart";

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
  color: string;
};

function formatNumber(value: number): string {
  return value.toFixed(3);
}

// Custom shape that renders a dot with error bars
function DotWithErrorBar(props: {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  payload?: ChartDataPoint;
  background?: { height?: number };
}) {
  const { x, y, width, payload, background } = props;
  if (
    x === undefined ||
    y === undefined ||
    width === undefined ||
    !payload ||
    !background?.height
  )
    return null;

  const cx = x + width / 2;
  const color = payload.color ?? CHART_COLORS[0];

  // Calculate error bar positions based on the chart scale
  // The background height represents the full Y range (0 to 1)
  const chartHeight = background.height;
  const [lowerError, upperError] = payload.error;

  // Convert error values to pixel offsets (chart is 0-1 scale)
  const lowerY = y + lowerError * chartHeight;
  const upperY = y - upperError * chartHeight;
  const errorBarWidth = 8;

  return (
    <g>
      {/* Vertical line */}
      <line
        x1={cx}
        y1={lowerY}
        x2={cx}
        y2={upperY}
        stroke={color}
        strokeWidth={2}
      />
      {/* Bottom cap */}
      <line
        x1={cx - errorBarWidth / 2}
        y1={lowerY}
        x2={cx + errorBarWidth / 2}
        y2={lowerY}
        stroke={color}
        strokeWidth={2}
      />
      {/* Top cap */}
      <line
        x1={cx - errorBarWidth / 2}
        y1={upperY}
        x2={cx + errorBarWidth / 2}
        y2={upperY}
        stroke={color}
        strokeWidth={2}
      />
      {/* Center dot */}
      <circle cx={cx} cy={y} r={5} fill={color} />
    </g>
  );
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

    // Sort alphabetically first to assign consistent colors across pages
    const sortedAlphabetically = [...entries].sort(([a], [b]) =>
      a.localeCompare(b),
    );

    // Create color map based on alphabetical order
    const colorMap = new Map(
      sortedAlphabetically.map(([name], index) => [
        name,
        CHART_COLORS[index % CHART_COLORS.length],
      ]),
    );

    // Sort by mean estimate (descending - best first) for display
    const sortedByMean = entries.sort(([, a], [, b]) => {
      const aMean = a?.mean_est ?? 0;
      const bMean = b?.mean_est ?? 0;
      return bMean - aMean;
    });

    return sortedByMean.map(([name, summary]): ChartDataPoint => {
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
        color: colorMap.get(name) ?? CHART_COLORS[0],
      };
    });
  }, [data.variant_summaries]);

  // Calculate max count for bar chart domain and evenly spaced ticks
  const { maxCount, countTicks } = useMemo(() => {
    if (chartData.length === 0)
      return { maxCount: 100, countTicks: [0, 25, 50, 75, 100] };
    const max = Math.max(...chartData.map((d) => d.count));
    const paddedMax = Math.ceil(max * 1.15); // Add 15% padding for labels
    // Generate 5 evenly spaced tick values
    const tickCount = 5;
    const ticks = Array.from({ length: tickCount }, (_, i) =>
      Math.round((paddedMax * i) / (tickCount - 1)),
    );
    return { maxCount: paddedMax, countTicks: ticks };
  }, [chartData]);

  if (chartData.length === 0) {
    return (
      <div className="text-fg-muted py-4 text-center text-sm">
        No variant data available
      </div>
    );
  }

  const chartMargin = { top: 10, right: 20, left: 10, bottom: 5 };

  return (
    <div className="flex flex-col gap-0">
      {/* Top chart: Mean performance with confidence intervals */}
      <div className="text-fg-secondary mb-1 text-xs font-medium">
        Mean Performance by Variant
      </div>
      <ChartContainer config={{}} className="h-[200px] w-full">
        <BarChart data={chartData} margin={chartMargin}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--color-border)"
            strokeOpacity={0.8}
          />
          <XAxis
            dataKey="name"
            tick={false}
            axisLine={{
              stroke: "var(--color-border)",
              strokeDasharray: "3 3",
              strokeOpacity: 0.8,
            }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={formatNumber}
            tick={{ fontSize: 10, fill: "var(--color-fg-tertiary)" }}
            axisLine={{
              stroke: "var(--color-border)",
              strokeDasharray: "3 3",
              strokeOpacity: 0.8,
            }}
            tickLine={{ stroke: "var(--color-border)", strokeOpacity: 0.8 }}
            width={50}
          />
          <ChartTooltip content={<PerformanceTooltip />} />
          <ReferenceLine
            y={0.5}
            stroke="var(--border)"
            strokeDasharray="3 3"
            opacity={0.5}
          />
          <Bar
            dataKey="mean"
            shape={<DotWithErrorBar />}
            isAnimationActive={false}
          />
        </BarChart>
      </ChartContainer>

      {/* Bottom chart: Number of evaluations */}
      <div className="text-fg-secondary mb-1 text-xs font-medium">
        Number of Evaluations by Variant
      </div>
      <ChartContainer config={{}} className="h-[260px] w-full">
        <BarChart data={chartData} margin={chartMargin}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--color-border)"
            strokeOpacity={0.8}
            vertical={false}
          />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 10, fill: "var(--color-fg-tertiary)" }}
            axisLine={{
              stroke: "var(--color-border)",
              strokeDasharray: "3 3",
              strokeOpacity: 0.8,
            }}
            tickLine={{ stroke: "var(--color-border)", strokeOpacity: 0.8 }}
            angle={-45}
            textAnchor="end"
            height={60}
            interval={0}
          />
          <YAxis
            domain={[0, maxCount]}
            ticks={countTicks}
            tick={{ fontSize: 10, fill: "var(--color-fg-tertiary)" }}
            axisLine={{
              stroke: "var(--color-border)",
              strokeDasharray: "3 3",
              strokeOpacity: 0.8,
            }}
            tickLine={{ stroke: "var(--color-border)", strokeOpacity: 0.8 }}
            width={50}
          />
          <ChartTooltip content={<CountTooltip />} />
          <Bar
            dataKey="count"
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
            label={{
              position: "top",
              fontSize: 9,
              fill: "var(--color-fg-tertiary)",
            }}
          >
            {chartData.map((entry) => (
              <Cell key={entry.name} fill={entry.color} fillOpacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
    </div>
  );
}
