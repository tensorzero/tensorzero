import { v7 as uuid } from "uuid";
import type {
  ParsedDatasetRow,
  ParsedChatInferenceDatapointRow,
  ParsedJsonInferenceDatapointRow,
} from "~/utils/clickhouse/datasets";
import {
  getDatasetCounts,
  staleDatapoint,
} from "~/utils/clickhouse/datasets.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { resolvedInputToTensorZeroInput } from "~/routes/api/tensorzero/inference.utils";
import type { Datapoint } from "~/utils/tensorzero";
import { toDatasetUrl } from "~/utils/urls";

// ============================================================================
// Transformation Functions
// ============================================================================

function transformOutputForTensorZero(
  output: ParsedDatasetRow["output"],
): string | null {
  if (output === null || output === undefined) {
    return null;
  } else if ("raw" in output) {
    if (output.raw === null) {
      return null;
    }
    try {
      return JSON.parse(output.raw);
    } catch (error) {
      throw new Error(
        `Invalid JSON in output.raw: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  } else if (typeof output === "object") {
    try {
      return JSON.parse(JSON.stringify(output));
    } catch (error) {
      throw new Error(
        `Failed to serialize output object: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  } else {
    return output;
  }
}

/**
 * Transforms a chat datapoint for submission to the TensorZero client.
 * Generates a new UUID and marks the datapoint as custom.
 */
function transformChatDatapoint(datapoint: ParsedChatInferenceDatapointRow): Datapoint {
  const transformed: Datapoint = {
    function_name: datapoint.function_name,
    id: datapoint.id,
    episode_id: datapoint.episode_id,
    input: resolvedInputToTensorZeroInput(datapoint.input),
    output: transformOutputForTensorZero(datapoint.output),
    tags: datapoint.tags || {},
    auxiliary: datapoint.auxiliary,
    source_inference_id: datapoint.source_inference_id,
    is_custom: datapoint.is_custom,
    name: datapoint.name,
    tool_params: datapoint.tool_params,
    staled_at: datapoint.staled_at,
  };
  return transformed;
}

/**
 * Transforms a JSON datapoint for submission to the TensorZero client.
 * Generates a new UUID and marks the datapoint as custom.
 */
function transformJsonDatapoint(datapoint: ParsedJsonInferenceDatapointRow): Datapoint {
  const transformed: Datapoint = {
    function_name: datapoint.function_name,
    id: datapoint.id,
    episode_id: datapoint.episode_id,
    input: resolvedInputToTensorZeroInput(datapoint.input),
    output: transformOutputForTensorZero(datapoint.output),
    tags: datapoint.tags || {},
    auxiliary: datapoint.auxiliary,
    source_inference_id: datapoint.source_inference_id,
    is_custom: datapoint.is_custom,
    name: datapoint.name,
    output_schema: datapoint.output_schema,
    staled_at: datapoint.staled_at,
  };
  return transformed;
}

// ============================================================================
// Core Operations
// ============================================================================

export async function deleteDatapoint(params: {
  dataset_name: string;
  id: string;
  functionType: "chat" | "json";
}): Promise<{ redirectTo: string }> {
  const { dataset_name, id, functionType } = params;

  await staleDatapoint(dataset_name, id, functionType);

  const datasetCounts = await getDatasetCounts({});
  const datasetCount = datasetCounts.find(
    (count) => count.dataset_name === dataset_name,
  );

  if (datasetCount === undefined) {
    return { redirectTo: "/datasets" };
  }
  return { redirectTo: toDatasetUrl(dataset_name) };
}

/**
 * Saves a datapoint by creating a new version with a new ID and marking the old one as stale.
 * The function type (chat/json) is automatically determined from the datapoint structure.
 *
 * TODO(#3765): remove this logic and use Rust logic instead, either via napi-rs or by calling an API server.
 */
export async function saveDatapoint(params: {
  parsedFormData: ParsedDatasetRow;
  functionType: "chat" | "json";
}): Promise<{ newId: string; }> {
  const { parsedFormData, functionType } = params;

  // Determine function type from datapoint structure and transform accordingly
  let datapoint: Datapoint;
  if (functionType === "json") {
    if (!("output_schema" in parsedFormData)) {
      throw new Error(`Json datapoint is missing output_schema`);
    }
    datapoint = transformJsonDatapoint(parsedFormData as ParsedJsonInferenceDatapointRow);
  } else if (functionType === "chat") {
    datapoint = transformChatDatapoint(parsedFormData as ParsedChatInferenceDatapointRow);
  } else {
    throw new Error(`Invalid function type: ${functionType}`);
  }

  // When saving a datapoint as new, we create a new ID, and mark the data point as "custom".
  datapoint.id = uuid();
  datapoint.is_custom = true;
  datapoint.episode_id = null;
  datapoint.staled_at = null;

  // For future reference:
  // These two calls would be a transaction but ClickHouse isn't transactional.
  //
  // TODO(shuyangli): this should actually use "createDatapoint" since we're creating a new datapoint. We should reason about the
  // safety and do it in a follow-up.
  const { id } = await getTensorZeroClient().updateDatapoint(
    parsedFormData.dataset_name,
    datapoint,
  );

  await staleDatapoint(
    parsedFormData.dataset_name,
    parsedFormData.id,
    functionType,
  );

  return { newId: id };
}
