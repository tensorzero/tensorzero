import { getDatasetMetadata } from "~/utils/clickhouse/datasets.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { toDatasetUrl } from "~/utils/urls";
import type {
  UpdateDatapointsMetadataRequest,
  UpdateDatapointRequest,
  ContentBlockChatOutput,
  JsonInferenceOutput,
} from "~/types/tensorzero";
import type { UpdateDatapointFormData } from "./formDataUtils";

// ============================================================================
// Transformation Functions
// ============================================================================

/**
 * Converts UpdateDatapointFormData to UpdateDatapointRequest for the API.
 */
function convertUpdateDatapointFormDataToUpdateDatapointRequest(
  formData: Omit<UpdateDatapointFormData, "action">,
  functionType: "chat" | "json",
): UpdateDatapointRequest {
  // TODO: this logic could be more type safe but it's progress...
  switch (functionType) {
    case "json": {
      return {
        type: "json",
        id: formData.id,
        input: formData.input,
        output: formData.output as JsonInferenceOutput | undefined,
        tags: formData.tags,
      };
    }

    case "chat": {
      return {
        type: "chat",
        id: formData.id,
        input: formData.input,
        output: formData.output as ContentBlockChatOutput[] | undefined,
        tags: formData.tags,
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
 * Updates a datapoint by creating a new version with a new ID and marking the old one as stale.
 * The function type (chat/json) is automatically determined from the datapoint structure.
 *
 * The v1 updateDatapoint endpoint automatically handles:
 * - Creating a new datapoint with a new v7 UUID
 * - Marking the old datapoint as stale (setting staled_at timestamp)
 * - Returning the new datapoint ID
 *
 * TODO(#3765): remove this logic and use Rust logic instead, either via napi-rs or by calling an API server.
 */
export async function updateDatapoint(params: {
  parsedFormData: Omit<UpdateDatapointFormData, "action">;
  functionType: "chat" | "json";
}): Promise<{ newId: string }> {
  const { parsedFormData, functionType } = params;

  // Convert the parsed form data to an UpdateDatapointRequest
  const updateRequest = convertUpdateDatapointFormDataToUpdateDatapointRequest(
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
        name,
      },
    ],
  };

  await getTensorZeroClient().updateDatapointsMetadata(
    datasetName,
    updateRequest,
  );
}
