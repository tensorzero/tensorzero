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
import { TimeGranularitySelector } from "~/components/function/variant/TimeGranularitySelector";
import { useTimeGranularityParam } from "~/hooks/use-time-granularity-param";
import { transformFeedbackTimeseries } from "~/components/function/variant/FeedbackSamplesTimeseries";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";

interface FunctionExperimentationProps {
  functionConfig: FunctionConfig;
  functionName: string;
  optimalProbabilities?: Record<string, number>;
  feedbackTimeseries?: CumulativeFeedbackTimeSeriesPoint[];
}

function extractVariantWeights(
  functionConfig: FunctionConfig,
  optimalProbabilities?: Record<string, number>,
): VariantWeight[] {
  const experimentationConfig = functionConfig.experimentation;

  let variantWeights: VariantWeight[];

  if (experimentationConfig.type === "static_weights") {
    // Extract candidate variants and their weights
    const candidateVariants = experimentationConfig.candidate_variants;
    variantWeights = Object.entries(candidateVariants)
      .filter(([, weight]) => weight !== undefined)
      .map(([variant_name, weight]) => ({
        variant_name,
        weight: weight!,
      }))
      .sort((a, b) => a.variant_name.localeCompare(b.variant_name));
  } else if (experimentationConfig.type === "uniform") {
    // For uniform distribution, all variants get equal weight
    const variantNames = Object.keys(functionConfig.variants);
    const equalWeight = 1.0 / variantNames.length;
    variantWeights = variantNames.map((variant_name) => ({
      variant_name,
      weight: equalWeight,
    }));
  } else if (experimentationConfig.type === "track_and_stop") {
    // For track_and_stop, use optimal probabilities if available
    if (optimalProbabilities) {
      variantWeights = Object.entries(optimalProbabilities).map(
        ([variant_name, weight]) => ({
          variant_name,
          weight,
        }),
      );
    } else {
      // If no optimal probabilities yet (e.g., due to null variances or insufficient data),
      // show equal weights for all candidate variants (nursery phase)
      const candidateVariants = experimentationConfig.candidate_variants;
      const equalWeight = 1.0 / candidateVariants.length;
      variantWeights = candidateVariants.map((variant_name) => ({
        variant_name,
        weight: equalWeight,
      }));
    }
  } else {
    // Default case (shouldn't happen, but TypeScript requires it)
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
  optimalProbabilities,
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
    optimalProbabilities,
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

  return (
    <>
      <ExperimentationPieChart variantWeights={variantWeights} />
      {shouldShowTimeseries && (
        <div className="space-y-4">
          <TimeGranularitySelector
            time_granularity={timeGranularity}
            onTimeGranularityChange={onTimeGranularityChange}
            includeCumulative={false}
            includeMinute={false}
            includeHour={true}
          />
          <Tabs defaultValue="means" className="w-full">
            <TabsList>
              <TabsTrigger value="means">Mean Reward</TabsTrigger>
              <TabsTrigger value="counts">Feedback Counts</TabsTrigger>
            </TabsList>
            <TabsContent value="means">
              <FeedbackMeansTimeseries
                meansData={meansData}
                countsData={countsData}
                variantNames={variantNames}
                timeGranularity={timeGranularity}
              />
            </TabsContent>
            <TabsContent value="counts">
              <FeedbackCountsTimeseries
                countsData={countsData}
                variantNames={variantNames}
                timeGranularity={timeGranularity}
              />
            </TabsContent>
          </Tabs>
        </div>
      )}
    </>
  );
});
