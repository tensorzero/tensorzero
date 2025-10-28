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
  trackAndStopState?: unknown;
}

function extractVariantWeights(
  functionConfig: FunctionConfig,
  trackAndStopState?: unknown,
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
      }));
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
    if (!trackAndStopState) {
      return [];
    }

    const response = trackAndStopState as {
      display_probabilities?: Record<string, number>;
    };

    if (!response.display_probabilities) {
      return [];
    }

    variantWeights = Object.entries(response.display_probabilities).map(
      ([variant_name, weight]) => ({
        variant_name,
        weight,
      }),
    );
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
}: FunctionExperimentationProps) {
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

  return <ExperimentationPieChart variantWeights={variantWeights} />;
});
