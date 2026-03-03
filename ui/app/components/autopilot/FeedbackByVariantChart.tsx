import type {
  FeedbackByVariant,
  VariantPerformanceRow,
} from "~/types/tensorzero";
import {
  transformVariantPerformances,
  VariantPerformanceChart,
} from "~/components/function/variant/VariantPerformanceChart";

interface FeedbackByVariantChartProps {
  data: FeedbackByVariant[];
  metricName: string;
  functionName: string;
}

/**
 * Converts FeedbackByVariant[] (aggregate stats per variant) into
 * VariantPerformanceRow[] for rendering with the shared chart component.
 *
 * Computes stdev from variance and derives a 95% confidence interval.
 */
function toPerformanceRows(
  feedback: FeedbackByVariant[],
): VariantPerformanceRow[] {
  return feedback.map((f) => {
    const count = Number(f.count);
    const stdev =
      f.variance != null ? Math.sqrt(Math.max(0, f.variance)) : undefined;
    const ci_error =
      stdev != null && count > 0
        ? (1.96 * stdev) / Math.sqrt(count)
        : undefined;

    return {
      period_start: "1970-01-01T00:00:00Z",
      variant_name: f.variant_name,
      count,
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
