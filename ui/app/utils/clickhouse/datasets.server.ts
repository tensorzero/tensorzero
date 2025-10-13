import { getNativeDatabaseClient } from "../tensorzero/native_client.server";
import type {
  DatasetMetadata,
  DatasetQueryParams,
  DatasetDetailRow,
  GetDatasetMetadataParams,
  GetDatasetRowsParams,
  GetDatapointParams,
  Datapoint,
  AdjacentDatapointIds,
  ToolCallConfigDatabaseInsert,
} from "tensorzero-node";
import type {
  ParsedDatasetRow,
  ParsedChatInferenceDatapointRow,
  ParsedJsonInferenceDatapointRow,
} from "./datasets";
import { displayInputToInput } from "./common";
import { getConfig, getFunctionConfig } from "../config/index.server";

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

export async function getNumberOfDatasets(): Promise<number> {
  const dbClient = await getNativeDatabaseClient();
  return await dbClient.getNumberOfDatasets();
}

export async function getDatasetRows(
  params: GetDatasetRowsParams,
): Promise<DatasetDetailRow[]> {
  const dbClient = await getNativeDatabaseClient();
  return await dbClient.getDatasetRows(params);
}

export async function getDatapoint(
  params: GetDatapointParams,
): Promise<Datapoint> {
  const dbClient = await getNativeDatabaseClient();
  return await dbClient.getDatapoint(params);
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

export async function insertDatapoint(
  datapoint: ParsedDatasetRow,
): Promise<void> {
  const dbClient = await getNativeDatabaseClient();
  const input = displayInputToInput(datapoint.input);

  if ("tool_params" in datapoint) {
    // Chat inference datapoint
    const chatDatapoint = datapoint as ParsedChatInferenceDatapointRow;
    await dbClient.insertDatapoint({
      type: "chat",
      dataset_name: chatDatapoint.dataset_name,
      function_name: chatDatapoint.function_name,
      id: chatDatapoint.id,
      name: chatDatapoint.name,
      episode_id: chatDatapoint.episode_id,
      input,
      output: chatDatapoint.output,
      // TODO(shuyangli): Fix this type conversion. This was serialized and deserialized across TypeScript and Rust boundaries with different types.
      tool_params:
        chatDatapoint.tool_params as unknown as ToolCallConfigDatabaseInsert,
      tags: chatDatapoint.tags,
      auxiliary: chatDatapoint.auxiliary,
      staled_at: chatDatapoint.staled_at ?? undefined,
      source_inference_id: chatDatapoint.source_inference_id ?? undefined,
      is_custom: chatDatapoint.is_custom,
    });
  } else {
    // JSON inference datapoint
    const jsonDatapoint = datapoint as ParsedJsonInferenceDatapointRow;
    await dbClient.insertDatapoint({
      type: "json",
      dataset_name: jsonDatapoint.dataset_name,
      function_name: jsonDatapoint.function_name,
      id: jsonDatapoint.id,
      name: jsonDatapoint.name,
      episode_id: jsonDatapoint.episode_id,
      input,
      output: jsonDatapoint.output,
      output_schema: jsonDatapoint.output_schema,
      tags: jsonDatapoint.tags,
      auxiliary: jsonDatapoint.auxiliary,
      staled_at: jsonDatapoint.staled_at ?? undefined,
      source_inference_id: jsonDatapoint.source_inference_id ?? undefined,
      is_custom: jsonDatapoint.is_custom,
    });
  }
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
