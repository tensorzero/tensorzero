import type { ModelLatencyDatapoint, TimeWindow } from "tensorzero-node";
import {
  Line,
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import React, { useRef, useState, useMemo, useCallback } from "react";
import { Await } from "react-router";
import { TimeWindowSelector } from "~/components/ui/TimeWindowSelector";
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

type LatencyMetric = "response_time_ms" | "ttft_ms";

const CHART_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
] as const;

const MARGIN = { top: 12, right: 16, bottom: 28, left: 56 };

/** Find latency at a given percentile p for one model’s quantile arrays. */
function latencyAtPercentile(
  p: number, // 0..1
  quantiles: number[],
  latencies: Array<number | null>,
): number | null {
  // Guard: empty or all nulls
  if (!quantiles.length || !latencies.length) return null;

  // Find first index i such that quantiles[i] >= p
  let i = quantiles.findIndex((q) => q >= p);
  if (i === -1) i = quantiles.length - 1; // p beyond max -> clamp to last

  // Skip null values by walking backward/forward if needed
  const getVal = (idx: number) =>
    idx >= 0 && idx < latencies.length ? latencies[idx] : null;

  // Exact hit or first ≥p
  let hiIdx = i;
  let hi = getVal(hiIdx);
  // Move hi forward until non-null (rare but safe)
  while (hiIdx < latencies.length && (hi == null || hi <= 0)) {
    hiIdx++;
    hi = getVal(hiIdx);
  }

  // Low bound just before hi
  let loIdx = Math.max(0, hiIdx - 1);
  let lo = getVal(loIdx);
  while (loIdx >= 0 && (lo == null || lo <= 0)) {
    loIdx--;
    lo = getVal(loIdx);
  }

  // If we only have one side, return that
  if (hi != null && lo == null) return hi;
  if (lo != null && hi == null) return lo;
  if (lo == null && hi == null) return null;

  // If p exactly equals quantiles[hiIdx] or lo==hi, return hi
  const qHi = quantiles[Math.min(hiIdx, quantiles.length - 1)];
  const qLo = quantiles[Math.max(loIdx, 0)];
  if (p === qHi || lo === hi || qHi === qLo) return hi!;

  // Linear interpolate between (qLo, lo) and (qHi, hi)
  const t = (p - qLo) / (qHi - qLo);
  return lo! + t * (hi! - lo!);
}

/** Tooltip that uses hovered percentile, not Recharts payload. */
function CustomTooltipContent({
  hoveredPct,
  latencyData,
  selectedMetric,
  quantiles,
  colorFor,
}: {
  hoveredPct: number | null;
  latencyData: ModelLatencyDatapoint[];
  selectedMetric: LatencyMetric;
  quantiles: number[];
  colorFor: (modelName: string) => string;
}) {
  if (hoveredPct == null) return null;

  const rows = latencyData
    .map((m) => {
      const arr =
        selectedMetric === "response_time_ms"
          ? m.response_time_ms_quantiles
          : m.ttft_ms_quantiles;
      const value = latencyAtPercentile(hoveredPct, quantiles, arr);
      return value && value > 0
        ? { name: m.model_name, value, color: colorFor(m.model_name) }
        : null;
    })
    .filter(Boolean) as { name: string; value: number; color: string }[];

  if (!rows.length) return null;

  return (
    <div className="border-border/50 bg-background min-w-[10rem] rounded-lg border px-2.5 py-1.5 text-xs shadow-xl">
      <div className="flex items-center justify-between gap-2 pb-1">
        <span className="text-muted-foreground">Percentile</span>
        <span className="text-foreground font-mono font-medium tabular-nums">
          {(hoveredPct * 100).toFixed(2)}%
        </span>
      </div>
      <div className="border-border/50 border-t pt-1.5">
        {rows.map((r) => (
          <div key={r.name} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-2.5 rounded-[2px]"
                style={{ background: r.color, border: `1px solid ${r.color}` }}
              />
              <span className="text-muted-foreground">{r.name}</span>
            </div>
            <span className="text-foreground font-mono font-medium tabular-nums">
              {Math.round(r.value)}ms
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function LatencyECDFChart({
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

  const colorFor = useCallback(
    (name: string) =>
      CHART_COLORS[modelNames.indexOf(name) % CHART_COLORS.length],
    [modelNames],
  );

  // --- Horizontal hover state & mouse tracking ---
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoveredPct, setHoveredPct] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(
    null,
  );

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    // Compute inner plot height after margins
    const innerTop = rect.top + MARGIN.top;
    const innerBottom = rect.bottom - MARGIN.bottom;
    const innerHeight = innerBottom - innerTop;
    const y = e.clientY;
    // ratio from top (0) to bottom (1) within inner plotting area
    let r = (y - innerTop) / innerHeight;
    r = Math.max(0, Math.min(1, r));
    const pct = 1 - r; // y-axis grows upward: top = 1.0, bottom = 0.0

    setHoveredPct(pct);
    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const handleMouseLeave = () => {
    setHoveredPct(null);
    setMousePos(null);
  };

  return (
    <div
      ref={containerRef}
      className="relative h-80 w-full"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* Floating tooltip anchored to mouse, using hoveredPct */}
      {hoveredPct != null && mousePos && (
        <div
          className="pointer-events-none absolute z-10"
          style={{
            left: Math.min(
              mousePos.x + 14,
              (containerRef.current?.clientWidth ?? 0) - 180,
            ),
            top: Math.max(mousePos.y - 16, 0),
          }}
        >
          <CustomTooltipContent
            hoveredPct={hoveredPct}
            latencyData={latencyData}
            selectedMetric={selectedMetric}
            quantiles={quantiles}
            colorFor={colorFor}
          />
        </div>
      )}

      <ResponsiveContainer>
        <LineChart data={data} margin={MARGIN}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="latency"
            tickLine={false}
            tickMargin={10}
            axisLine={true}
            tickFormatter={(v) => `${v}ms`}
          />
          <YAxis
            domain={[0, 1]}
            tickLine={false}
            tickMargin={10}
            axisLine={true}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />

          {/* Horizontal hover guideline */}
          {hoveredPct != null && (
            <ReferenceLine
              y={hoveredPct}
              stroke="#666666"
              strokeDasharray="3 3"
              strokeWidth={2}
              ifOverflow="extendDomain"
            />
          )}

          {/* Legend: you can keep your existing ChartLegend if desired */}

          {modelNames.map((name) => (
            <Line
              key={name}
              type="stepAfter"
              dataKey={name}
              name={name}
              stroke={colorFor(name)}
              strokeWidth={2}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* --- Your transform stays the same; included here for completeness --- */
type ECDFDataPoint = {
  latency: number;
  [modelName: string]: number | null;
};

function transformLatencyData(
  latencyData: ModelLatencyDatapoint[],
  selectedMetric: LatencyMetric,
  quantiles: number[],
): { data: ECDFDataPoint[]; modelNames: string[] } {
  const modelNames = latencyData.map((d) => d.model_name);
  const allLatencyValues = new Set<number>();

  latencyData.forEach((modelData) => {
    const arr =
      selectedMetric === "response_time_ms"
        ? modelData.response_time_ms_quantiles
        : modelData.ttft_ms_quantiles;
    arr.forEach((v) => {
      if (v != null && v > 0) allLatencyValues.add(v);
    });
  });

  const xs = Array.from(allLatencyValues).sort((a, b) => a - b);
  const data: ECDFDataPoint[] = [];

  xs.forEach((latency) => {
    const dp: ECDFDataPoint = { latency };
    modelNames.forEach((name) => {
      const md = latencyData.find((d) => d.model_name === name)!;
      const arr =
        selectedMetric === "response_time_ms"
          ? md.response_time_ms_quantiles
          : md.ttft_ms_quantiles;
      // Highest quantile idx where value <= latency
      let idx = -1;
      for (let i = 0; i < arr.length; i++) {
        const qv = arr[i];
        if (qv != null && qv > 0 && qv <= latency) idx = i;
      }
      dp[name] = idx >= 0 ? quantiles[idx] : 0;
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
            Empirical cumulative distribution function (eCDF) of latency metrics
            by model
          </CardDescription>
        </div>
        <div className="flex flex-col justify-center gap-2">
          <TimeWindowSelector
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
              <LatencyECDFChart
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
