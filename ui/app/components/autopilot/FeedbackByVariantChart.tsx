import type {
  FeedbackByVariant,
  VariantPerformanceRow,
} from "~/types/tensorzero";
import {
  transformVariantPerformances,
  VariantPerformanceChart,
} from "~/components/function/variant/VariantPerformanceChart";

// JSON.parse returns number (not bigint) for count, so override it.
type ParsedFeedbackByVariant = Omit<FeedbackByVariant, "count"> & {
  count: number;
};

interface FeedbackByVariantChartProps {
  data: ParsedFeedbackByVariant[];
  metricName: string;
  functionName: string;
}

/**
 * Validates that a parsed JSON value looks like a FeedbackByVariant entry.
 * Guards against malformed tool result data rendering NaN bars.
 */
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

/**
 * Validates that a parsed JSON array contains FeedbackByVariant entries.
 */
export function parseFeedbackByVariant(
  data: unknown,
): ParsedFeedbackByVariant[] | null {
  if (!Array.isArray(data)) return null;
  if (!data.every(isFeedbackByVariant)) return null;
  return data;
}

export interface FeedbackChartData {
  data: ParsedFeedbackByVariant[];
  metricName: string;
  functionName: string;
}

/**
 * Extracts chart data from a successful get_feedback_by_variant tool result.
 * Returns null if the tool result is not a valid feedback chart payload.
 */
export function parseFeedbackChartData(
  toolResult: string,
  toolArguments: unknown,
): FeedbackChartData | null {
  try {
    const raw: unknown = JSON.parse(toolResult);
    const data = parseFeedbackByVariant(raw);
    if (!data) return null;
    const params =
      typeof toolArguments === "object" && toolArguments !== null
        ? (toolArguments as Record<string, unknown>)
        : null;
    const metricName =
      typeof params?.metric_name === "string" ? params.metric_name : "metric";
    const functionName =
      typeof params?.function_name === "string"
        ? params.function_name
        : "function";
    return { data, metricName, functionName };
  } catch {
    return null;
  }
}

/**
 * Converts FeedbackByVariant[] (aggregate stats per variant) into
 * VariantPerformanceRow[] for rendering with the shared chart component.
 *
 * Computes stdev from variance and derives a 95% confidence interval.
 */
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
      period_start: "1970-01-01T00:00:00Z",
      variant_name: f.variant_name,
      count: f.count,
      avg_metric: f.mean,
      stdev,
      ci_error,
    };
  });
}

export default function FeedbackByVariantChart({
  data,
  metricName,
  functionName,
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
    <div className="flex flex-col gap-1">
      <div className="text-fg-secondary text-xs font-medium">
        <code>{metricName}</code> for <code>{functionName}</code>
      </div>
      <VariantPerformanceChart
        data={chartData}
        variantNames={variantNames}
        timeGranularity="cumulative"
        singleVariantMode={singleVariantMode}
      />
    </div>
  );
}
