import { getNativeDatabaseClient } from "../tensorzero/native_client.server";
import type {
  DatasetMetadata,
  DatasetQueryParams,
  DatasetDetailRow,
  GetDatasetMetadataParams,
  GetDatapointParams,
  ListDatapointsRequest,
  Datapoint,
  AdjacentDatapointIds,
} from "tensorzero-node";
import type {
  ParsedDatasetRow,
  ParsedChatInferenceDatapointRow,
  ParsedJsonInferenceDatapointRow,
} from "./datasets";
import { getConfig, getFunctionConfig } from "../config/index.server";
import { resolveStoredInput } from "../resolve.server";

// TODO(shuyangli): Consider removing this file and fully use DatabaseClient from tensorzero-node/lib.

/**
 * Executes an INSERT INTO ... SELECT ... query to insert rows into the dataset table.
 *
 * The destination table is determined by the provided `inferenceType`:
 * - "chat"  → ChatInferenceDatapoint
 * - "json"  → JsonInferenceDatapoint
 *
 * This function calls the Rust implementation which builds and executes the query.
 * If there is a datapoint with the same `source_inference_id` and `function_name`
 * in the destination table, it will be skipped.
 *
 * @param params - The dataset query parameters.
 * @returns The number of rows inserted.
 */
export async function insertRowsForDataset(
  params: DatasetQueryParams,
): Promise<number> {
  const dbClient = await getNativeDatabaseClient();
  return await dbClient.insertRowsForDataset(params);
}

/**
 * Executes a COUNT query for the dataset rows matching the provided parameters.
 *
 * Note: This function does not support LIMIT or OFFSET.
 *
 * @param params - The query parameters.
 * @returns A promise resolving to the count of rows matching the query.
 */
export async function countRowsForDataset(
  params: DatasetQueryParams,
): Promise<number> {
  const dbClient = await getNativeDatabaseClient();
  return await dbClient.countRowsForDataset(params);
}

/*
Get name and count for all datasets.
This function should sum the counts of chat and json inferences for each dataset.
The groups should be ordered by last_updated in descending order.
*/
export async function getDatasetMetadata(
  params: GetDatasetMetadataParams,
): Promise<DatasetMetadata[]> {
  const dbClient = await getNativeDatabaseClient();
  return await dbClient.getDatasetMetadata(params);
}

export async function countDatasets(): Promise<number> {
  const dbClient = await getNativeDatabaseClient();
  return await dbClient.countDatasets();
}

/// TODO (GabrielBianconi): This method currently retrieves the entire datapoint.
/// For performance reasons, consider implementing a version that only returns what we need for the table view.
/// `input` and `output` can be especially large.
export async function listDatapoints(
  dataset_name: string,
  params: ListDatapointsRequest,
): Promise<DatasetDetailRow[]> {
  const dbClient = await getNativeDatabaseClient();
  const response = await dbClient.listDatapoints(dataset_name, params);

  // Convert full Datapoint objects to DatasetDetailRow summaries
  return response.datapoints.map((dp) => ({
    id: dp.id,
    type: dp.type,
    function_name: dp.function_name,
    name: dp.name,
    episode_id: dp.episode_id,
    updated_at: dp.updated_at,
  }));
}

export async function getDatapoint(
  params: GetDatapointParams,
): Promise<ParsedDatasetRow | null> {
  // Swallow the error and return null if the datapoint is not found, to preserve existing behavior.
  const dbClient = await getNativeDatabaseClient();
  let datapoint: Datapoint | null = null;
  try {
    datapoint = await dbClient.getDatapoint(params);
    // TODO(shuyangli): Rename ParsedDatasetRow to be more clear.
  } catch (error) {
    if (
      error instanceof Error &&
      error.message.includes("Datapoint not found")
    ) {
      return null;
    }
    throw error;
  }
  return await datapointToParsedDatasetRow(datapoint);
}

export async function staleDatapoint(
  dataset_name: string,
  datapoint_id: string,
  function_type: "chat" | "json",
): Promise<void> {
  const dbClient = await getNativeDatabaseClient();
  await dbClient.staleDatapoint({
    dataset_name,
    datapoint_id,
    function_type,
  });
}

export async function countDatapointsForDatasetFunction(
  dataset_name: string,
  function_name: string,
): Promise<number | null> {
  const config = await getConfig();
  const functionConfig = await getFunctionConfig(function_name, config);
  const function_type = functionConfig?.type;
  if (!function_type) {
    return null;
  }
  const dbClient = await getNativeDatabaseClient();
  return dbClient.countDatapointsForDatasetFunction({
    dataset_name,
    function_name,
    function_type,
  });
}

export async function getAdjacentDatapointIds(
  dataset_name: string,
  datapoint_id: string,
): Promise<AdjacentDatapointIds> {
  const dbClient = await getNativeDatabaseClient();
  return await dbClient.getAdjacentDatapointIds({
    dataset_name,
    datapoint_id,
  });
}

/**
 * Converts a backend Datapoint to a frontend ParsedDatasetRow.
 * This is used when receiving data from the backend API.
 * TODO(shuyangli): Remove soon!
 */
export async function datapointToParsedDatasetRow(
  datapoint: Datapoint,
): Promise<ParsedDatasetRow> {
  const resolvedInput = await resolveStoredInput(datapoint.input);

  if (datapoint.type === "chat") {
    const chatDatapoint: ParsedChatInferenceDatapointRow = {
      dataset_name: datapoint.dataset_name,
      function_name: datapoint.function_name,
      id: datapoint.id,
      name: datapoint.name,
      episode_id: datapoint.episode_id ?? null,
      input: resolvedInput,
      output: datapoint.output,
      tags: datapoint.tags ?? null,
      auxiliary: datapoint.auxiliary,
      is_deleted: datapoint.is_deleted,
      is_custom: datapoint.is_custom,
      staled_at: datapoint.staled_at ?? null,
      source_inference_id: datapoint.source_inference_id ?? null,
      updated_at: datapoint.updated_at,
    };
    return chatDatapoint;
  } else {
    const jsonDatapoint: ParsedJsonInferenceDatapointRow = {
      dataset_name: datapoint.dataset_name,
      function_name: datapoint.function_name,
      id: datapoint.id,
      name: datapoint.name,
      episode_id: datapoint.episode_id ?? null,
      input: resolvedInput,
      output: datapoint.output,
      output_schema: datapoint.output_schema,
      tags: datapoint.tags ?? null,
      auxiliary: datapoint.auxiliary,
      is_deleted: datapoint.is_deleted,
      is_custom: datapoint.is_custom,
      staled_at: datapoint.staled_at ?? null,
      source_inference_id: datapoint.source_inference_id ?? null,
      updated_at: datapoint.updated_at,
    };
    return jsonDatapoint;
  }
}
