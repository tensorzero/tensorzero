import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type {
  FunctionConfig,
  InferenceCountByVariant,
} from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";

export type VariantCountWithMetadata = Omit<
  InferenceCountByVariant,
  "inference_count"
> & {
  inference_count: number;
  type: string;
  weight: number | null;
};

export interface VariantsSectionData {
  variant_counts: VariantCountWithMetadata[];
}

export async function fetchVariantsSectionData(params: {
  function_name: string;
  function_config: FunctionConfig;
}): Promise<VariantsSectionData> {
  const { function_name, function_config } = params;

  const client = getTensorZeroClient();
  const variant_counts = await client.getInferenceCount(function_name, {
    groupBy: "variant",
  });

  const observedVariants = new Set<string>();
  const variant_counts_with_metadata = (
    variant_counts.count_by_variant ?? []
  ).map((variant_count) => {
    observedVariants.add(variant_count.variant_name);
    const variant_config = function_config.variants[variant_count.variant_name];
    return {
      ...variant_count,
      inference_count: Number(variant_count.inference_count),
      type:
        function_name === DEFAULT_FUNCTION
          ? "chat_completion"
          : (variant_config?.inner.type ?? "unknown"),
      weight:
        function_name === DEFAULT_FUNCTION
          ? null
          : (variant_config?.inner.weight ?? null),
    };
  });

  // Add configured variants that have no inferences yet
  if (function_name !== DEFAULT_FUNCTION) {
    for (const [variant_name, variant_config] of Object.entries(
      function_config.variants,
    )) {
      if (!observedVariants.has(variant_name)) {
        variant_counts_with_metadata.push({
          variant_name,
          inference_count: 0,
          last_used_at: "",
          type: variant_config.inner.type,
          weight: variant_config.inner.weight,
        });
      }
    }
  }

  return { variant_counts: variant_counts_with_metadata };
}
