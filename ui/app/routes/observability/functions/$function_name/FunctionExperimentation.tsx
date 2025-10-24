import type {
  CumulativeFeedbackTimeSeriesPoint,
  FunctionConfig,
  TimeWindow,
} from "tensorzero-node";
import {
  ExperimentationPieChart,
  type VariantWeight,
} from "~/components/experimentation/PieChart";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { memo } from "react";
import { FeedbackSamplesTimeseries } from "~/components/function/variant/FeedbackSamplesTimeseries";

interface FunctionExperimentationProps {
  functionConfig: FunctionConfig;
  functionName: string;
  optimalProbabilities?: Record<string, number>;
  feedbackTimeseries?: CumulativeFeedbackTimeSeriesPoint[];
  feedback_time_granularity?: TimeWindow;
  onCumulativeFeedbackTimeGranularityChange?: (granularity: TimeWindow) => void;
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
  feedback_time_granularity,
  onCumulativeFeedbackTimeGranularityChange,
}: FunctionExperimentationProps) {
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

  return (
    <>
      <ExperimentationPieChart variantWeights={variantWeights} />
      {functionConfig.experimentation.type === "track_and_stop" &&
        feedbackTimeseries &&
        feedbackTimeseries.length > 0 &&
        feedback_time_granularity &&
        onCumulativeFeedbackTimeGranularityChange && (
          <FeedbackSamplesTimeseries
            feedbackTimeseries={feedbackTimeseries}
            time_granularity={feedback_time_granularity}
            onCumulativeFeedbackTimeGranularityChange={
              onCumulativeFeedbackTimeGranularityChange
            }
          />
        )}
    </>
  );
});
