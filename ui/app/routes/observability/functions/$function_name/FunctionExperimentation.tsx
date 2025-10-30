import type {
  CumulativeFeedbackTimeSeriesPoint,
  FunctionConfig,
} from "tensorzero-node";
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
        <ExperimentationPieChart variantWeights={variantWeights} />
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
            />
          </TabsContent>
        </>
      )}
    </Tabs>
  );
});
