import { v7 as uuid } from "uuid";
import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";
import {
  getDatasetCounts,
  staleDatapoint,
} from "~/utils/clickhouse/datasets.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { resolvedInputToTensorZeroInput } from "~/routes/api/tensorzero/inference.utils";
import type { Datapoint } from "~/utils/tensorzero";

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
  return { redirectTo: `/datasets/${dataset_name}` };
}

export async function saveDatapoint(params: {
  parsedFormData: ParsedDatasetRow;
  functionType: "chat" | "json";
}): Promise<{ newId: string }> {
  const { parsedFormData, functionType } = params;

  // Transform input to match TensorZero client's expected format
  const transformedInput = resolvedInputToTensorZeroInput(parsedFormData.input);
  const transformedOutput = transformOutputForTensorZero(parsedFormData.output);

  // For future reference:
  // These two calls would be a transaction but ClickHouse isn't transactional.
  const baseDatapoint = {
    function_name: parsedFormData.function_name,
    input: transformedInput,
    output: transformedOutput,
    tags: parsedFormData.tags || {},
    auxiliary: parsedFormData.auxiliary,
    is_custom: true, // we're saving it after an edit, so it's custom
    source_inference_id: parsedFormData.source_inference_id,
    id: uuid(), // We generate a new ID here because we want old evaluation runs to be able to point to the correct data.
  };

  let datapoint: Datapoint;
  if (functionType === "json" && "output_schema" in parsedFormData) {
    datapoint = {
      ...baseDatapoint,
      output_schema: parsedFormData.output_schema,
    };
  } else if (functionType === "chat") {
    datapoint = {
      ...baseDatapoint,
      tool_params:
        "tool_params" in parsedFormData
          ? parsedFormData.tool_params
          : undefined,
    };
  } else {
    throw new Error(
      `Unexpected function type "${functionType}" or missing required properties on datapoint`,
    );
  }

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
