import type { FunctionConfig } from "tensorzero-node";
import {
  ExperimentationPieChart,
  type VariantWeight,
} from "~/components/experimentation/PieChart";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { memo } from "react";

interface FunctionExperimentationProps {
  functionConfig: FunctionConfig;
  functionName: string;
  optimalProbabilities?: Record<string, number>;
}

function extractVariantWeights(
  functionConfig: FunctionConfig,
  optimalProbabilities?: Record<string, number>,
): VariantWeight[] {
  const experimentationConfig = functionConfig.experimentation;

  if (experimentationConfig.type === "static_weights") {
    // Extract candidate variants and their weights
    const candidateVariants = experimentationConfig.candidate_variants;
    return Object.entries(candidateVariants)
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
    return variantNames.sort().map((variant_name) => ({
      variant_name,
      weight: equalWeight,
    }));
  } else if (experimentationConfig.type === "track_and_stop") {
    // For track_and_stop, use optimal probabilities if available
    if (optimalProbabilities) {
      return Object.entries(optimalProbabilities)
        .map(([variant_name, weight]) => ({
          variant_name,
          weight,
        }))
        .sort((a, b) => a.variant_name.localeCompare(b.variant_name));
    }
    // If no optimal probabilities yet (e.g., due to null variances or insufficient data),
    // show equal weights for all candidate variants (nursery phase)
    const candidateVariants = experimentationConfig.candidate_variants;
    const equalWeight = 1.0 / candidateVariants.length;
    return candidateVariants.sort().map((variant_name) => ({
      variant_name,
      weight: equalWeight,
    }));
  }

  // Default case (shouldn't happen, but TypeScript requires it)
  return [];
}

const FunctionExperimentation = memo(function FunctionExperimentation({
  functionConfig,
  functionName,
  optimalProbabilities,
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

  return <ExperimentationPieChart variantWeights={variantWeights} />;
});

export default FunctionExperimentation;
