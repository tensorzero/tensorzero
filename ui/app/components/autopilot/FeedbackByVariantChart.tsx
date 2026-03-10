import type {
  FeedbackByVariant,
  VariantPerformanceRow,
} from "~/types/tensorzero";
import {
  transformVariantPerformances,
  VariantPerformanceChart,
} from "~/components/function/variant/VariantPerformanceChart";

// JSON.parse returns number (not bigint) for count, so override it.
export type ParsedFeedbackByVariant = Omit<FeedbackByVariant, "count"> & {
  count: number;
};

// Guards against malformed tool result data rendering NaN bars.
function isFeedbackByVariant(v: unknown): v is ParsedFeedbackByVariant {
  if (typeof v !== "object" || v === null) return false;
  const obj = v as Record<string, unknown>;
  return (
    typeof obj.variant_name === "string" &&
    typeof obj.mean === "number" &&
    typeof obj.count === "number" &&
    (obj.variance == null || typeof obj.variance === "number")
  );
}

export function parseFeedbackByVariant(
  data: unknown,
): ParsedFeedbackByVariant[] | null {
  if (!Array.isArray(data) || data.length === 0) return null;
  if (!data.every(isFeedbackByVariant)) return null;
  return data;
}

export function parseFeedbackChartData(
  toolResult: string,
): ParsedFeedbackByVariant[] | null {
  try {
    const raw: unknown = JSON.parse(toolResult);
    return parseFeedbackByVariant(raw);
  } catch {
    return null;
  }
}

// Computes stdev from variance and derives a 95% confidence interval.
function toPerformanceRows(
  feedback: ParsedFeedbackByVariant[],
): VariantPerformanceRow[] {
  return feedback.map((f) => {
    const stdev =
      f.variance != null ? Math.sqrt(Math.max(0, f.variance)) : undefined;
    const ci_error =
      stdev != null && f.count > 0
        ? (1.96 * stdev) / Math.sqrt(f.count)
        : undefined;

    return {
      // Dummy date — displayed as "All time" via timeGranularity="cumulative"
      period_start: "1970-01-01T00:00:00Z",
      variant_name: f.variant_name,
      count: f.count,
      avg_metric: f.mean,
      stdev,
      ci_error,
    };
  });
}

interface FeedbackByVariantChartProps {
  data: ParsedFeedbackByVariant[];
}

export default function FeedbackByVariantChart({
  data,
}: FeedbackByVariantChartProps) {
  const rows = toPerformanceRows(data);
  const { data: chartData, variantNames } = transformVariantPerformances(rows);

  if (chartData.length === 0) {
    return (
      <div className="text-fg-muted py-4 text-center text-sm">
        No variant performance data available
      </div>
    );
  }

  const singleVariantMode = variantNames.length === 1;

  return (
    <VariantPerformanceChart
      data={chartData}
      variantNames={variantNames}
      timeGranularity="cumulative"
      singleVariantMode={singleVariantMode}
    />
  );
}
