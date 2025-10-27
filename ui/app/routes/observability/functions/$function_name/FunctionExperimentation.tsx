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
  displayProbabilities?: Record<string, number>;
}

function extractVariantWeights(
  displayProbabilities?: Record<string, number>,
): VariantWeight[] {
  // All probability computation is done in Rust via get_display_sampling_probabilities()
  if (!displayProbabilities) {
    return [];
  }

  const variantWeights = Object.entries(displayProbabilities).map(
    ([variant_name, weight]) => ({
      variant_name,
      weight,
    }),
  );

  // Sort alphabetically for consistent display order (affects pie chart segment order and reload stability)
  return variantWeights.sort((a, b) =>
    a.variant_name.localeCompare(b.variant_name),
  );
}

export const FunctionExperimentation = memo(function FunctionExperimentation({
  functionName,
  displayProbabilities,
}: FunctionExperimentationProps) {
  // Don't render experimentation section for the default function
  if (functionName === DEFAULT_FUNCTION) {
    return null;
  }

  const variantWeights = extractVariantWeights(displayProbabilities);

  // Don't render if there are no variant weights
  if (variantWeights.length === 0) {
    return null;
  }

  return <ExperimentationPieChart variantWeights={variantWeights} />;
});
