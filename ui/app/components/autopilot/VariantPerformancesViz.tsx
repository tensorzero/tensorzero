import type { VariantPerformancesVisualization } from "~/types/tensorzero";
import { TIME_WINDOWS, type TimeWindow } from "~/utils/date";
import {
  transformVariantPerformances,
  VariantPerformanceChart,
} from "~/components/function/variant/VariantPerformanceChart";
import { logger } from "~/utils/logger";

type VariantPerformancesVizProps = {
  data: VariantPerformancesVisualization;
};

const VALID_TIME_WINDOWS: Set<string> = new Set(TIME_WINDOWS);

export default function VariantPerformancesViz({
  data,
}: VariantPerformancesVizProps) {
  let timeGranularity: TimeWindow;
  if (VALID_TIME_WINDOWS.has(data.time_granularity)) {
    timeGranularity = data.time_granularity as TimeWindow;
  } else {
    logger.warn(
      `Unknown time_granularity "${data.time_granularity}", falling back to "week"`,
    );
    timeGranularity = "week";
  }
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
