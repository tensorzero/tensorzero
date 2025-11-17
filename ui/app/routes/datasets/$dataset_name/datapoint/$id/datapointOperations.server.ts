import type {
  ParsedDatasetRow,
  ParsedChatInferenceDatapointRow,
  ParsedJsonInferenceDatapointRow,
} from "~/utils/clickhouse/datasets";
import { getDatasetMetadata } from "~/utils/clickhouse/datasets.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { resolvedInputToTensorZeroInput } from "~/routes/api/tensorzero/inference.utils";
import { toDatasetUrl } from "~/utils/urls";
import type {
  UpdateDatapointsMetadataRequest,
  UpdateDatapointRequest,
  Input,
  ContentBlockChatOutput,
  DynamicToolParams,
  JsonDatapointOutputUpdate,
  JsonValue,
} from "~/types/tensorzero";

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
 * Converts a parsed dataset row to an UpdateDatapointRequest.
 *
 * TODO: This is a temporary function while we migrate more of the UI to the canonical binding types.
 */
function convertParsedDatasetRowToUpdateDatapointRequest(
  parsedFormData: ParsedDatasetRow,
  functionType: "chat" | "json",
): UpdateDatapointRequest {
  switch (functionType) {
    case "json": {
      const datapoint = parsedFormData as ParsedJsonInferenceDatapointRow;
      return {
        type: "json",
        id: datapoint.id,
        input: resolvedInputToTensorZeroInput(datapoint.input) as Input,
        output: datapoint.output
          ? ({
              raw: JSON.stringify(
                transformOutputForTensorZero(datapoint.output),
              ),
            } as JsonDatapointOutputUpdate)
          : undefined,
        output_schema: datapoint.output_schema as JsonValue,
        tags: datapoint.tags || undefined,
      };
    }

    case "chat": {
      const datapoint = parsedFormData as ParsedChatInferenceDatapointRow;
      return {
        type: "chat",
        id: datapoint.id,
        input: resolvedInputToTensorZeroInput(datapoint.input) as Input,
        output: datapoint.output as ContentBlockChatOutput[] | undefined,
        tags: datapoint.tags || undefined,
        tool_params: datapoint.tool_params as DynamicToolParams,
      };
    }
  }
}

// ============================================================================
// Core Operations
// ============================================================================

export async function deleteDatapoint(params: {
  dataset_name: string;
  id: string;
}): Promise<{ redirectTo: string }> {
  const { dataset_name, id } = params;

  await getTensorZeroClient().deleteDatapoints(dataset_name, [id]);

  const datasetCounts = await getDatasetMetadata({});
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
 * The v1 updateDatapoint endpoint automatically handles:
 * - Creating a new datapoint with a new v7 UUID
 * - Marking the old datapoint as stale (setting staled_at timestamp)
 * - Returning the new datapoint ID
 *
 * TODO(#3765): remove this logic and use Rust logic instead, either via napi-rs or by calling an API server.
 */
export async function saveDatapoint(params: {
  parsedFormData: ParsedDatasetRow;
  functionType: "chat" | "json";
}): Promise<{ newId: string }> {
  const { parsedFormData, functionType } = params;

  // Convert the parsed form data to an UpdateDatapointRequest
  const updateRequest = convertParsedDatasetRowToUpdateDatapointRequest(
    parsedFormData,
    functionType,
  );

  // The updateDatapoint endpoint will automatically generate a new ID and stale the old one
  const { id } = await getTensorZeroClient().updateDatapoint(
    parsedFormData.dataset_name,
    updateRequest,
  );

  return { newId: id };
}

/**
 * Renames a datapoint by calling the update datapoints metadata endpoint.
 *
 * Arguments:
 * - `datasetName`: the name of the dataset
 * - `datapointId`: the ID of the datapoint to rename
 * - `newName`: the new name of the datapoint (or `null` explicitly to unset)
 */
export async function renameDatapoint(params: {
  datasetName: string;
  datapointId: string;
  name: string | null;
}): Promise<void> {
  const { datasetName, datapointId, name } = params;

  const updateRequest: UpdateDatapointsMetadataRequest = {
    datapoints: [
      {
        id: datapointId,
        metadata: {
          name,
        },
      },
    ],
  };

  await getTensorZeroClient().updateDatapointsMetadata(
    datasetName,
    updateRequest,
  );
}
