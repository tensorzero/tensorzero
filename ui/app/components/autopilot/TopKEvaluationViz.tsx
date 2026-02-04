import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Cell,
  Customized,
  ReferenceLine,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { ChartContainer, ChartTooltip } from "~/components/ui/chart";
import { Markdown } from "~/components/ui/markdown";
import type { TopKEvaluationVisualization } from "~/types/tensorzero";
import { CHART_COLORS, formatChartNumber } from "~/utils/chart";

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
  failed: boolean;
};

const FAILED_OPACITY = 0.35;
const FAILED_BACKGROUND_COLOR = "var(--color-bg-tertiary)";
const MAX_LABEL_LENGTH = 10;
const NUM_DECIMAL_PLACES = 3;

function formatNumber(value: number): string {
  return value.toFixed(NUM_DECIMAL_PLACES);
}

function truncateLabel(label: string): string {
  if (label.length <= MAX_LABEL_LENGTH) return label;
  return label.slice(0, MAX_LABEL_LENGTH - 1) + "…";
}

// Custom X-axis tick that truncates long labels and shows full name on hover
function TruncatedTick({
  x,
  y,
  payload,
}: {
  x?: number;
  y?: number;
  payload?: { value: string };
}) {
  if (x === undefined || y === undefined || !payload) return null;

  const fullName = payload.value;
  const displayName = truncateLabel(fullName);

  return (
    <g transform={`translate(${x},${y})`}>
      <title>{fullName}</title>
      <text
        x={0}
        y={0}
        dy={4}
        textAnchor="end"
        transform="rotate(-45)"
        style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}
        fill="currentColor"
      >
        {displayName}
      </text>
    </g>
  );
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
  const opacity = payload.failed ? FAILED_OPACITY : 1;

  // Calculate error bar positions based on the chart scale
  // The background height represents the full Y range (0 to 1)
  const chartHeight = background.height;
  const [lowerError, upperError] = payload.error;

  // Convert error values to pixel offsets (chart is 0-1 scale)
  const lowerY = y + lowerError * chartHeight;
  const upperY = y - upperError * chartHeight;
  const errorBarWidth = 8;

  return (
    <g opacity={opacity}>
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
    <div className="border-border/50 bg-background rounded-md border p-2 text-xs shadow-lg">
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
      {data.failed && (
        <div className="mt-1 font-medium text-red-500">⚠️ Failed</div>
      )}
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
    <div className="border-border/50 bg-background rounded-md border p-2 text-xs shadow-lg">
      <div className="mb-1 font-medium">{data.name}</div>
      <div className="text-fg-secondary">
        Samples:{" "}
        <span className="font-mono">{data.count.toLocaleString()}</span>
      </div>
      {data.failed && (
        <div className="mt-1 font-medium text-red-500">⚠️ Failed</div>
      )}
    </div>
  );
}

// Custom component to render background for failed variants
type FailedBackgroundsProps = {
  chartData: ChartDataPoint[];
  offset?: { top: number; left: number; width: number; height: number };
};

function FailedVariantBackgrounds({
  chartData,
  offset,
}: FailedBackgroundsProps & Record<string, unknown>) {
  if (!offset || chartData.length === 0) {
    return null;
  }

  const failedIndices = chartData
    .map((d, i) => (d.failed ? i : -1))
    .filter((i) => i >= 0);

  if (failedIndices.length === 0) {
    return null;
  }

  const barWidth = offset.width / chartData.length;

  return (
    <g className="failed-backgrounds">
      {failedIndices.map((index) => {
        const x = offset.left + index * barWidth;
        return (
          <rect
            key={index}
            x={x}
            y={offset.top}
            width={barWidth}
            height={offset.height}
            fill={FAILED_BACKGROUND_COLOR}
          />
        );
      })}
    </g>
  );
}

// Custom component to render warning icons below the x-axis for failed variants
type FailedWarningIconsProps = {
  chartData: ChartDataPoint[];
  offset?: { top: number; left: number; width: number; height: number };
};

function FailedWarningIcons({
  chartData,
  offset,
}: FailedWarningIconsProps & Record<string, unknown>) {
  if (!offset || chartData.length === 0) {
    return null;
  }

  const failedIndices = chartData
    .map((d, i) => (d.failed ? i : -1))
    .filter((i) => i >= 0);

  if (failedIndices.length === 0) {
    return null;
  }

  const barWidth = offset.width / chartData.length;
  const yPosition = offset.top + offset.height + 16;

  return (
    <g className="failed-warning-icons">
      {failedIndices.map((index) => {
        const cx = offset.left + index * barWidth + barWidth / 2;
        return (
          <text
            key={index}
            x={cx}
            y={yPosition}
            textAnchor="middle"
            fontSize={12}
          >
            ⚠️
          </text>
        );
      })}
    </g>
  );
}

// Custom component to render horizontal separation lines
type SeparationLinesProps = {
  separationYValues: Array<{ k: number; y: number }>;
  chartData: ChartDataPoint[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  yAxisMap?: any;
  offset?: { top: number; left: number; width: number };
};

function SeparationLines({
  separationYValues,
  chartData,
  yAxisMap,
  offset,
}: SeparationLinesProps & Record<string, unknown>) {
  if (!yAxisMap || !offset || separationYValues.length === 0) {
    return null;
  }

  const yAxis = Object.values(yAxisMap)[0] as {
    scale: (value: number) => number;
  };

  if (!yAxis?.scale) {
    return null;
  }

  // Calculate where non-failed variants end
  const nonFailedCount = chartData.filter((d) => !d.failed).length;
  const barWidth = offset.width / chartData.length;

  const xLeft = offset.left;
  // End line at the boundary between non-failed and failed variants
  const xRight = offset.left + nonFailedCount * barWidth;

  return (
    <g className="separation-lines">
      {separationYValues.map(({ k, y }) => {
        const yPixel = yAxis.scale(y);

        return (
          <g key={k}>
            {/* Dashed horizontal line */}
            <line
              x1={xLeft}
              y1={yPixel}
              x2={xRight}
              y2={yPixel}
              stroke="var(--color-fg-tertiary)"
              strokeWidth={1.5}
              strokeDasharray="4 4"
            />
            {/* Label inside chart, above the line on the left */}
            <text
              x={xLeft + 4}
              y={yPixel - 4}
              textAnchor="start"
              fontSize={10}
              fill="var(--color-fg-secondary)"
              fontWeight={500}
            >
              top {k}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function SummaryText({ text }: { text: string }) {
  return (
    <div className="border-border bg-bg-tertiary mt-4 rounded-md border p-4">
      <div className="text-fg-secondary mb-2 text-xs font-medium">
        Analysis Summary
      </div>
      <Markdown className="prose prose-sm max-w-none text-sm">{text}</Markdown>
    </div>
  );
}

export default function TopKEvaluationViz({ data }: TopKEvaluationVizProps) {
  const chartData = useMemo(() => {
    const entries = Object.entries(data.variant_summaries);

    // Separate failed and non-failed variants
    const nonFailedEntries = entries.filter(([, summary]) => !summary?.failed);
    const failedEntries = entries.filter(([, summary]) => summary?.failed);

    // Sort all variants alphabetically to assign consistent colors across pages
    const sortedAlphabetically = [...entries].sort(([a], [b]) =>
      a.localeCompare(b),
    );

    // Create color map based on alphabetical order (for all variants)
    const colorMap = new Map(
      sortedAlphabetically.map(([name], index) => [
        name,
        CHART_COLORS[index % CHART_COLORS.length],
      ]),
    );

    // Sort non-failed by mean estimate (descending - best first)
    const sortedNonFailed = nonFailedEntries.sort(([, a], [, b]) => {
      const aMean = a?.mean_est ?? 0;
      const bMean = b?.mean_est ?? 0;
      return bMean - aMean;
    });

    // Sort failed by mean estimate (descending) - displayed on the right
    const sortedFailed = failedEntries.sort(([, a], [, b]) => {
      const aMean = a?.mean_est ?? 0;
      const bMean = b?.mean_est ?? 0;
      return bMean - aMean;
    });

    // Combine: non-failed first, then failed
    const sortedEntries = [...sortedNonFailed, ...sortedFailed];

    return sortedEntries.map(([name, summary]): ChartDataPoint => {
      const mean = summary?.mean_est ?? 0;
      const lower = summary?.cs_lower ?? 0;
      const upper = summary?.cs_upper ?? 0;
      // Convert bigint to number for chart rendering
      const count = Number(summary?.count ?? 0);
      const failed = summary?.failed ?? false;

      return {
        name,
        mean,
        lower,
        upper,
        // Error bar values are relative to the mean: [lower error, upper error]
        error: [mean - lower, upper - mean],
        count,
        color: colorMap.get(name) ?? CHART_COLORS[0],
        failed,
      };
    });
  }, [data.variant_summaries]);

  // Compute Y values for separation lines
  // Lines are drawn at the midpoint between:
  // - The min lower bound of the top-k variants (sorted by lower bound descending)
  // - The max upper bound of the remaining variants
  // Note: Failed variants are excluded from this calculation
  const separationYValues = useMemo(() => {
    const separationIndices = data.confident_top_k_sizes ?? [];
    if (separationIndices.length === 0) return [];

    // Sort non-failed variants by lower bound descending (matching the backend algorithm)
    const sortedByLower = Object.entries(data.variant_summaries)
      .filter(([, summary]) => !summary?.failed)
      .map(([name, summary]) => ({
        name,
        lower: summary?.cs_lower ?? 0,
        upper: summary?.cs_upper ?? 0,
      }))
      .sort((a, b) => b.lower - a.lower);

    return separationIndices
      .map((k) => {
        if (k <= 0 || k >= sortedByLower.length) return null;

        // Top k variants (indices 0 to k-1)
        const topK = sortedByLower.slice(0, k);
        // Remaining variants (indices k to end)
        const rest = sortedByLower.slice(k);

        // Min lower bound of top k
        const minLowerTopK = Math.min(...topK.map((v) => v.lower));
        // Max upper bound of rest
        const maxUpperRest = Math.max(...rest.map((v) => v.upper));

        // Y value is midpoint between separation
        const y = (minLowerTopK + maxUpperRest) / 2;

        return { k, y };
      })
      .filter((v): v is { k: number; y: number } => v !== null);
  }, [data.variant_summaries, data.confident_top_k_sizes]);

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
        <BarChart accessibilityLayer data={chartData} margin={chartMargin}>
          <Customized
            component={(props: Omit<FailedBackgroundsProps, "chartData">) => (
              <FailedVariantBackgrounds {...props} chartData={chartData} />
            )}
          />
          <CartesianGrid vertical={false} />
          <XAxis dataKey="name" tick={false} axisLine={true} tickLine={false} />
          <YAxis
            domain={[0, 1]}
            tickFormatter={formatNumber}
            tickLine={false}
            tickMargin={10}
            axisLine={true}
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
          <Customized
            component={(
              props: Omit<
                SeparationLinesProps,
                "separationYValues" | "chartData"
              >,
            ) => (
              <SeparationLines
                {...props}
                separationYValues={separationYValues}
                chartData={chartData}
              />
            )}
          />
          <Customized
            component={(props: Omit<FailedWarningIconsProps, "chartData">) => (
              <FailedWarningIcons {...props} chartData={chartData} />
            )}
          />
        </BarChart>
      </ChartContainer>

      {/* Bottom chart: Number of evaluations */}
      <div className="text-fg-secondary mb-1 text-xs font-medium">
        Number of Evaluations by Variant
      </div>
      <ChartContainer config={{}} className="h-[260px] w-full">
        <BarChart accessibilityLayer data={chartData} margin={chartMargin}>
          <Customized
            component={(props: Omit<FailedBackgroundsProps, "chartData">) => (
              <FailedVariantBackgrounds {...props} chartData={chartData} />
            )}
          />
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="name"
            tick={<TruncatedTick />}
            axisLine={true}
            tickLine={false}
            tickMargin={10}
            height={70}
            interval={0}
          />
          <YAxis
            domain={[0, maxCount]}
            ticks={countTicks}
            tickLine={false}
            tickMargin={10}
            axisLine={true}
            tickFormatter={(value) => formatChartNumber(Number(value))}
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
              <Cell
                key={entry.name}
                fill={entry.color}
                fillOpacity={entry.failed ? FAILED_OPACITY : 1}
              />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>

      {data.summary_text && <SummaryText text={data.summary_text} />}
    </div>
  );
}
