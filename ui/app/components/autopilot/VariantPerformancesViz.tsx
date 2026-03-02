import type {
  VariantPerformancesVisualization,
  TimeWindow,
} from "~/types/tensorzero";
import {
  transformVariantPerformances,
  VariantPerformanceChart,
} from "~/components/function/variant/VariantPerformance";

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

  return (
    <div className="flex flex-col gap-1">
      <div className="text-fg-secondary text-xs font-medium">
        <code>{data.metric_name}</code> for <code>{data.function_name}</code>
      </div>
      <VariantPerformanceChart
        data={chartData}
        variantNames={variantNames}
        timeGranularity={timeGranularity}
        singleVariantMode={singleVariantMode}
      />
    </div>
  );
}
