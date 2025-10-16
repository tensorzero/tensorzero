import z from "zod";
import { getClickhouseClient } from "./client.server";
import { getNativeDatabaseClient } from "../tensorzero/native_client.server";
import type {
  DatasetMetadata,
  DatasetQueryParams,
  DatasetDetailRow,
  GetDatasetMetadataParams,
  GetDatasetRowsParams,
} from "tensorzero-node";
import {
  DatapointRowSchema,
  type DatapointRow,
  type ParsedDatasetRow,
  ParsedChatInferenceDatapointRowSchema,
  ParsedJsonInferenceDatapointRowSchema,
} from "./datasets";
import type { AdjacentIds } from "./inference";
import { adjacentIdsSchema } from "./inference";
import {
  contentBlockChatOutputSchema,
  CountSchema,
  displayInputToInput,
  inputSchema,
  jsonInferenceOutputSchema,
} from "./common";
import { getConfig, getFunctionConfig } from "../config/index.server";
import { resolveInput } from "../resolve.server";
import { logger } from "~/utils/logger";

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
  const resultSet = await getClickhouseClient().query({
    query: `
      SELECT
        toUInt32(uniqExact(dataset_name)) as count
      FROM (
        SELECT dataset_name
        FROM ChatInferenceDatapoint FINAL
        WHERE staled_at IS NULL
        UNION ALL
        SELECT dataset_name
        FROM JsonInferenceDatapoint FINAL
        WHERE staled_at IS NULL
      )
    `,
    format: "JSONEachRow",
  });
  const rows = await resultSet.json<{ count: number }>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}

export async function getDatasetRows(
  params: GetDatasetRowsParams,
): Promise<DatasetDetailRow[]> {
  const dbClient = await getNativeDatabaseClient();
  return await dbClient.getDatasetRows(params);
}

export async function getDatapoint(
  dataset_name: string,
  id: string,
  allow_stale: boolean = false,
): Promise<ParsedDatasetRow | null> {
  let chat_query = `
    SELECT
      dataset_name,
      function_name,
      id,
      name,
      episode_id,
      input,
      output,
      tool_params,
      tags,
      auxiliary,
      source_inference_id,
      is_custom,
      formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at,
      formatDateTime(staled_at, '%Y-%m-%dT%H:%i:%SZ') as staled_at
    FROM ChatInferenceDatapoint FINAL
    WHERE dataset_name = {dataset_name:String}
      AND id = {id:String}
  `;
  if (!allow_stale) {
    chat_query += "\nAND staled_at IS NULL";
  }

  let json_query = `
    SELECT
      dataset_name,
      function_name,
      id,
      name,
      episode_id,
      input,
      output,
      output_schema,
      tags,
      auxiliary,
      source_inference_id,
      is_custom,
      formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at,
      formatDateTime(staled_at, '%Y-%m-%dT%H:%i:%SZ') AS staled_at
    FROM JsonInferenceDatapoint FINAL
    WHERE dataset_name = {dataset_name:String}
      AND id = {id:String}
  `;
  if (!allow_stale) {
    json_query += "\nAND staled_at IS NULL";
  }

  const [chatResult, jsonResult] = await Promise.all([
    getClickhouseClient()
      .query({
        query: chat_query,
        format: "JSONEachRow",
        query_params: { dataset_name, id },
      })
      .then((rs) => rs.json<DatapointRow[]>()),
    getClickhouseClient()
      .query({
        query: json_query,
        format: "JSONEachRow",
        query_params: { dataset_name, id },
      })
      .then((rs) => rs.json<DatapointRow[]>()),
  ]);

  const allResults = [...chatResult, ...jsonResult];
  if (allResults.length === 0) {
    return null;
  }

  if (allResults.length > 1) {
    throw new Error(
      `Expected exactly one result for dataset ${dataset_name} and id ${id}, but found ${allResults.length}`,
    );
  }

  const row = DatapointRowSchema.parse(allResults[0]);
  const parsedRow = await parseDatapointRow(row);

  return parsedRow;
}

async function parseDatapointRow(row: DatapointRow): Promise<ParsedDatasetRow> {
  const parsedInput = inputSchema.parse(JSON.parse(row.input));
  const config = await getConfig();
  const functionConfig = await getFunctionConfig(row.function_name, config);
  const resolvedInput = await resolveInput(parsedInput, functionConfig);
  if ("tool_params" in row) {
    // Chat inference row
    const processedRow = {
      ...row,
      input: resolvedInput,
      output: row.output
        ? z.array(contentBlockChatOutputSchema).parse(JSON.parse(row.output))
        : undefined,
      tool_params:
        row.tool_params === ""
          ? undefined
          : z
              .record(z.string(), z.unknown())
              .parse(JSON.parse(row.tool_params)),
      tags: row.tags,
    };
    return ParsedChatInferenceDatapointRowSchema.parse(processedRow);
  } else {
    // JSON inference row
    const processedRow = {
      ...row,
      input: resolvedInput,
      output: row.output
        ? jsonInferenceOutputSchema.parse(JSON.parse(row.output))
        : undefined,
      output_schema: z
        .record(z.string(), z.unknown())
        .parse(JSON.parse(row.output_schema)),
    };
    return ParsedJsonInferenceDatapointRowSchema.parse(processedRow);
  }
}

export async function staleDatapoint(
  dataset_name: string,
  datapoint_id: string,
  function_type: "chat" | "json",
): Promise<void> {
  // Use the function type to determine which table to update
  const table =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";

  const query = `
    INSERT INTO {table:Identifier}
    (
      dataset_name,
      function_name,
      id,
      name,
      episode_id,
      input,
      output,
      ${function_type === "chat" ? "tool_params" : "output_schema"},
      tags,
      auxiliary,
      is_deleted,
      source_inference_id,
      is_custom,
      staled_at,
      updated_at
    )
    SELECT
      dataset_name,
      function_name,
      id,
      name,
      episode_id,
      input,
      output,
      ${function_type === "chat" ? "tool_params" : "output_schema"},
      tags,
      auxiliary,
      is_deleted,
      source_inference_id,
      is_custom,
      now64() as staled_at,
      now64() as updated_at
    FROM {table:Identifier} FINAL
    WHERE dataset_name = {dataset_name:String} AND id = {datapoint_id:String}
  `;

  try {
    await getClickhouseClient().query({
      query,
      query_params: {
        table,
        dataset_name,
        datapoint_id,
      },
    });
  } catch (error) {
    logger.error(`Error staling datapoint ${datapoint_id}:`, error);
    throw error;
  }
}

export async function insertDatapoint(
  datapoint: ParsedDatasetRow,
): Promise<void> {
  validateDatasetName(datapoint.dataset_name);
  const table =
    "tool_params" in datapoint
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";
  const input = displayInputToInput(datapoint.input);
  const values = [
    {
      dataset_name: datapoint.dataset_name,
      function_name: datapoint.function_name,
      id: datapoint.id,
      name: datapoint.name,
      episode_id: datapoint.episode_id,
      input: input,
      output: datapoint.output,
      tags: datapoint.tags,
      auxiliary: datapoint.auxiliary,
      is_deleted: false,
      // Add type-specific fields
      ...("tool_params" in datapoint
        ? { tool_params: datapoint.tool_params }
        : {}),
      ...("output_schema" in datapoint
        ? { output_schema: datapoint.output_schema }
        : {}),
      source_inference_id: datapoint.source_inference_id,
      is_custom: datapoint.is_custom,
    },
  ];

  await getClickhouseClient().insert({
    table,
    values,
    format: "JSONEachRow",
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
  const table =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";
  const resultSet = await getClickhouseClient().query({
    query: `SELECT toUInt32(count()) as count FROM {table:Identifier} WHERE dataset_name = {dataset_name:String} AND function_name = {function_name:String}`,
    format: "JSONEachRow",
    query_params: { dataset_name, function_name, table },
  });
  const rows = await resultSet.json<{ count: number }>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}

function validateDatasetName(dataset_name: string) {
  if (dataset_name === "builder" || dataset_name.startsWith("tensorzero::")) {
    throw new Error("Invalid dataset name");
  }
}

export async function getAdjacentDatapointIds(
  dataset_name: string,
  datapoint_id: string,
): Promise<AdjacentIds> {
  const query = `
    WITH DatasetIds AS (
      SELECT toUInt128(id) as id_uint FROM ChatInferenceDatapoint WHERE dataset_name = {dataset_name:String}
      UNION ALL
      SELECT toUInt128(id) as id_uint FROM JsonInferenceDatapoint WHERE dataset_name = {dataset_name:String}
    )
    SELECT
      NULLIF(
      (SELECT uint_to_uuid(min(id_uint)) FROM DatasetIds WHERE id_uint > toUInt128({datapoint_id:UUID})),
      toUUID('00000000-0000-0000-0000-000000000000')
      ) as next_id,
      NULLIF(
        (SELECT uint_to_uuid(max(id_uint)) FROM DatasetIds WHERE id_uint < toUInt128({datapoint_id:UUID})),
        toUUID('00000000-0000-0000-0000-000000000000')
      ) as previous_id
    FROM DatasetIds
  `;
  const resultSet = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { dataset_name, datapoint_id },
  });
  const rows = await resultSet.json<AdjacentIds>();
  const parsedRows = rows.map((row) => adjacentIdsSchema.parse(row));
  return parsedRows[0];
}
