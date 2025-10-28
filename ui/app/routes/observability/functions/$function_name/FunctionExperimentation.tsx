import type {
  CumulativeFeedbackTimeSeriesPoint,
  FunctionConfig,
  TrackAndStopResponse,
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
  trackAndStopState?: TrackAndStopResponse;
  feedbackTimeseries?: CumulativeFeedbackTimeSeriesPoint[];
}

function extractVariantWeights(
  functionConfig: FunctionConfig,
  trackAndStopState?: TrackAndStopResponse,
): VariantWeight[] {
  const experimentationConfig = functionConfig.experimentation;

  let variantWeights: VariantWeight[];

  if (experimentationConfig.type === "static_weights") {
    // Extract weights from config and normalize to probabilities
    const candidateVariants = experimentationConfig.candidate_variants;
    variantWeights = Object.entries(candidateVariants)
      .filter(([, weight]) => weight !== undefined)
      .map(([variant_name, weight]) => ({
        variant_name,
        weight: weight!,
      }))
      .sort((a, b) => a.variant_name.localeCompare(b.variant_name));
  } else if (experimentationConfig.type === "uniform") {
    // Compute equal probabilities for all variants
    const variantNames = Object.keys(functionConfig.variants);
    const equalWeight = 1.0 / variantNames.length;
    variantWeights = variantNames.map((variant_name) => ({
      variant_name,
      weight: equalWeight,
    }));
  } else if (experimentationConfig.type === "track_and_stop") {
    // Extract display probabilities from track-and-stop response
    // Always use the probabilities from computeTrackAndStopState()
    if (!trackAndStopState?.display_probabilities) {
      return [];
    }

    variantWeights = Object.entries(
      trackAndStopState.display_probabilities,
    ).map(([variant_name, weight]) => ({
      variant_name,
      weight,
    }));
  } else {
    variantWeights = [];
  }

  // Sort alphabetically for consistent display order (affects pie chart segment order and reload stability)
  return variantWeights.sort((a, b) =>
    a.variant_name.localeCompare(b.variant_name),
  );
}

export const FunctionExperimentation = memo(function FunctionExperimentation({
  functionConfig,
  functionName,
  trackAndStopState,
  feedbackTimeseries,
}: FunctionExperimentationProps) {
  const [timeGranularity, onTimeGranularityChange] = useTimeGranularityParam(
    "cumulative_feedback_time_granularity",
    "week",
  );

  // Don't render experimentation section for the default function
  if (functionName === DEFAULT_FUNCTION) {
    return null;
  }

  const variantWeights = extractVariantWeights(
    functionConfig,
    trackAndStopState,
  );

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
