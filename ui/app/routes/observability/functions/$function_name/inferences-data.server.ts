import { countInferencesForFunction } from "~/utils/clickhouse/inference.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { applyPaginationLogic } from "~/utils/pagination";

export type InferencesSectionData = Awaited<
  ReturnType<typeof fetchInferencesSectionData>
>;

export function countInferences(
  function_name: string,
  namespace: string | undefined,
) {
  const tag = namespace ? `tensorzero::namespace::${namespace}` : undefined;
  return countInferencesForFunction(function_name, tag);
}

export async function fetchInferencesSectionData(params: {
  function_name: string;
  beforeInference: string | null;
  afterInference: string | null;
  limit: number;
  namespace: string | undefined;
}) {
  const { function_name, beforeInference, afterInference, limit, namespace } =
    params;
  const tag = namespace ? `tensorzero::namespace::${namespace}` : undefined;

  const client = getTensorZeroClient();
  const inferenceResult = await client.listInferenceMetadata({
    function_name,
    before: beforeInference || undefined,
    after: afterInference || undefined,
    limit: limit + 1, // Fetch one extra to determine pagination
    tag,
  });

  const {
    items: inferences,
    hasNextPage: hasNextInferencePage,
    hasPreviousPage: hasPreviousInferencePage,
  } = applyPaginationLogic(inferenceResult.inference_metadata, limit, {
    before: beforeInference,
    after: afterInference,
  });

  return {
    inferences,
    hasNextInferencePage,
    hasPreviousInferencePage,
  };
}
