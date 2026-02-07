import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { FunctionConfig } from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";

export type VariantsSectionData = {
  variant_counts: {
    variant_name: string;
    inference_count: bigint;
    last_used_at: string;
    type: string;
    weight: number | null;
  }[];
};

export async function fetchVariantsSectionData(params: {
  function_name: string;
  function_config: FunctionConfig;
}): Promise<VariantsSectionData> {
  const { function_name, function_config } = params;

  const client = getTensorZeroClient();
  const variant_counts = await client.getInferenceCount(function_name, {
    groupBy: "variant",
  });

  const variant_counts_with_metadata = (
    variant_counts.count_by_variant ?? []
  ).map((variant_count) => {
    let variant_config = function_config.variants[
      variant_count.variant_name
    ] || {
      inner: {
        type: "unknown",
        weight: 0,
      },
    };

    if (function_name === DEFAULT_FUNCTION) {
      variant_config = {
        inner: {
          type: "chat_completion",
          model: variant_count.variant_name,
          weight: null,
          templates: {},
          temperature: null,
          top_p: null,
          max_tokens: null,
          presence_penalty: null,
          frequency_penalty: null,
          seed: null,
          stop_sequences: null,
          json_mode: null,
          retries: { num_retries: 0, max_delay_s: 0 },
        },
        timeouts: {
          non_streaming: { total_ms: null },
          streaming: { ttft_ms: null },
        },
      };
    }

    return {
      ...variant_count,
      type: variant_config.inner.type,
      weight: variant_config.inner.weight,
    };
  });

  return { variant_counts: variant_counts_with_metadata };
}
