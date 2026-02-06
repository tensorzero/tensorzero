import { countInferencesForFunction } from "~/utils/clickhouse/inference.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { applyPaginationLogic } from "~/utils/pagination";

export type InferencesSectionData = Awaited<
  ReturnType<typeof fetchInferencesSectionData>
>;

export function countInferences(function_name: string) {
  return countInferencesForFunction(function_name);
}

export async function fetchInferencesSectionData(params: {
  function_name: string;
  beforeInference: string | null;
  afterInference: string | null;
  limit: number;
  countPromise: Promise<number>;
}) {
  const {
    function_name,
    beforeInference,
    afterInference,
    limit,
    countPromise,
  } = params;

  const client = getTensorZeroClient();
  const inferencePromise = client.listInferenceMetadata({
    function_name,
    before: beforeInference || undefined,
    after: afterInference || undefined,
    limit: limit + 1, // Fetch one extra to determine pagination
  });
  const [inferenceResult, num_inferences] = await Promise.all([
    inferencePromise,
    countPromise,
  ]);

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
    num_inferences,
  };
}
