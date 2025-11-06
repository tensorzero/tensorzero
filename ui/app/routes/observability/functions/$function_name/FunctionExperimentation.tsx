import type {
  CumulativeFeedbackTimeSeriesPoint,
  FunctionConfig,
} from "~/types/tensorzero";
import {
  ExperimentationPieChart,
  type VariantWeight,
} from "~/components/experimentation/PieChart";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { memo } from "react";
import { FeedbackCountsTimeseries } from "~/components/function/variant/FeedbackCountsTimeseries";
import { FeedbackMeansTimeseries } from "~/components/function/variant/FeedbackMeansTimeseries";
import { useTimeGranularityParam } from "~/hooks/use-time-granularity-param";
import { transformFeedbackTimeseries } from "~/components/function/variant/FeedbackSamplesTimeseries";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { CHART_COLORS } from "~/utils/chart";

interface FunctionExperimentationProps {
  functionConfig: FunctionConfig;
  functionName: string;
  feedbackTimeseries?: CumulativeFeedbackTimeSeriesPoint[];
  variantSamplingProbabilities: Record<string, number>;
}

export const FunctionExperimentation = memo(function FunctionExperimentation({
  functionConfig,
  functionName,
  feedbackTimeseries,
  variantSamplingProbabilities,
}: FunctionExperimentationProps) {
  const [timeGranularity, onTimeGranularityChange] = useTimeGranularityParam(
    "cumulative_feedback_time_granularity",
    "week",
  );

  // Don't render experimentation section for the default function
  if (functionName === DEFAULT_FUNCTION) {
    return null;
  }

  // Convert the probabilities from the loader to VariantWeight format
  const variantWeights: VariantWeight[] = Object.entries(
    variantSamplingProbabilities,
  )
    .map(([variant_name, weight]) => ({
      variant_name,
      weight,
    }))
    .sort((a, b) => a.variant_name.localeCompare(b.variant_name));

  // Don't render if there are no variant weights
  if (variantWeights.length === 0) {
    return null;
  }

  // Transform feedback timeseries data once for both charts
  const shouldShowTimeseries =
    functionConfig.experimentation.type === "track_and_stop" &&
    feedbackTimeseries &&
    feedbackTimeseries.length > 0;

  const { countsData, meansData, variantNames } = shouldShowTimeseries
    ? transformFeedbackTimeseries(feedbackTimeseries!, timeGranularity)
    : { countsData: [], meansData: [], variantNames: [] };

  // Extract metric name for track_and_stop experimentation
  const metricName =
    functionConfig.experimentation.type === "track_and_stop"
      ? functionConfig.experimentation.metric
      : "";

  // Create a centralized chart config to ensure consistent colors across all panels
  // Use union of variant names from current weights and historical feedback data
  // to handle recently disabled variants that still appear in historical timeseries
  const allVariantNames = new Set([
    ...variantWeights.map((v) => v.variant_name),
    ...variantNames,
  ]);
  const sortedVariantNames = Array.from(allVariantNames).sort((a, b) =>
    a.localeCompare(b),
  );

  const chartConfig: Record<string, { label: string; color: string }> =
    sortedVariantNames.reduce(
      (config, variantName, index) => ({
        ...config,
        [variantName]: {
          label: variantName,
          color: CHART_COLORS[index % CHART_COLORS.length],
        },
      }),
      {},
    );

  return (
    <Tabs defaultValue="weights" className="w-full">
      {shouldShowTimeseries && (
        <TabsList>
          <TabsTrigger value="weights">Variant Weights</TabsTrigger>
          <TabsTrigger value="means">Estimated Performance</TabsTrigger>
          <TabsTrigger value="counts">Feedback Count</TabsTrigger>
        </TabsList>
      )}
      <TabsContent value="weights">
        <ExperimentationPieChart
          variantWeights={variantWeights}
          chartConfig={chartConfig}
        />
      </TabsContent>
      {shouldShowTimeseries && (
        <>
          <TabsContent value="means">
            <FeedbackMeansTimeseries
              meansData={meansData}
              countsData={countsData}
              variantNames={variantNames}
              timeGranularity={timeGranularity}
              metricName={metricName}
              time_granularity={timeGranularity}
              onTimeGranularityChange={onTimeGranularityChange}
              chartConfig={chartConfig}
            />
          </TabsContent>
          <TabsContent value="counts">
            <FeedbackCountsTimeseries
              countsData={countsData}
              variantNames={variantNames}
              timeGranularity={timeGranularity}
              metricName={metricName}
              time_granularity={timeGranularity}
              onTimeGranularityChange={onTimeGranularityChange}
              chartConfig={chartConfig}
            />
          </TabsContent>
        </>
      )}
    </Tabs>
  );
});
