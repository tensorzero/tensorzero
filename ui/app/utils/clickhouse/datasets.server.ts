import z from "zod";
import { getClickhouseClient } from "./client.server";
import {
  DatasetCountInfoSchema,
  DatasetDetailRowSchema,
  DatapointInsertSchema,
  DatapointRowSchema,
  DatasetQueryParamsSchema,
  type DatasetCountInfo,
  type DatasetDetailRow,
  type DatapointInsert,
  type DatapointRow,
  type DatasetQueryParams,
  type ParsedDatasetRow,
  ParsedChatInferenceDatapointRowSchema,
  ParsedJsonInferenceDatapointRowSchema,
} from "./datasets";
import type { AdjacentIds } from "./inference";
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
 * Constructs a SELECT query for either the Chat or JSON dataset table.
 *
 * The query is built by:
 * - Choosing the appropriate table based on `inferenceType`.
 * - Constructing the SELECT field list (with adjustments if joining demonstrations).
 * - Appending optional JOIN clauses for metric filtering or demonstration feedback.
 * - Adding WHERE conditions based on `function_name`, `variant_name`, and any extra clauses.
 * - Applying LIMIT and OFFSET if provided.
 *
 * If `include_output` is false (and no demonstration join is used), the output field is replaced with NULL.
 *
 * @param params - The query parameters.
 * @returns An object containing the constructed query and its parameters.
 */
function buildDatasetSelectQuery(params: DatasetQueryParams): {
  query: string;
  query_params: Record<string, string | number>;
} {
  const {
    inferenceType,
    function_name,
    variant_name,
    extra_where,
    extra_params,
    metric_filter,
    output_source,
    limit,
    offset,
  } = params;

  // Validate: if variant_name is provided, function_name must also be provided.
  if (variant_name && !function_name) {
    throw new Error(
      "If variant_name is provided, function_name must also be provided.",
    );
  }

  // Select the appropriate table based on inference type.
  const tableName =
    inferenceType === "chat" ? "ChatInference" : "JsonInference";

  // Build the list of fields to select.
  let selectFields: string[];
  if (inferenceType === "chat") {
    selectFields = [
      "function_name",
      "id",
      "episode_id",
      // When building a dataset from inferences, there are no datapoint names.
      "NULL as name",
      "input",
      "output",
      "tool_params",
      "tags",
    ];
  } else {
    selectFields = [
      "function_name",
      "id",
      "episode_id",
      // When building a dataset from inferences, there are no datapoint names.
      "NULL as name",
      "input",
      "output",
      "output_schema",
      "tags",
    ];
  }

  // Adjust the output field based on flags:
  // - If join_demonstrations is true, use the demonstration's value as output.
  // - Otherwise, if include_output is false, replace output with NULL.
  if (output_source === "demonstration") {
    selectFields = selectFields.map((field) =>
      field === "output" ? "demo.value as output" : field,
    );
  } else if (output_source === "none") {
    selectFields = selectFields.map((field) =>
      field === "output" ? "NULL AS output" : field,
    );
  }

  // Always include an auxiliary field (currently an empty string).
  selectFields.push("'' AS auxiliary");

  // Start building the base query.
  let query = `SELECT ${selectFields.join(", ")} FROM ${tableName}`;

  // Prepare WHERE clause array and query parameters object.
  const whereClauses: string[] = [];
  const queryParams: Record<string, string | number> = {};

  // Merge any extra parameters into the query parameters.
  if (extra_params) {
    Object.assign(queryParams, extra_params);
  }

  // Add condition for function_name if provided.
  if (function_name) {
    whereClauses.push("function_name = {function_name:String}");
    queryParams.function_name = function_name;
  }

  // Add condition for variant_name if provided.
  if (variant_name) {
    whereClauses.push("variant_name = {variant_name:String}");
    queryParams.variant_name = variant_name;
  }

  // -------------------------------------------------------------------
  // Metric Filter Join Logic:
  // If a metric_filter is provided, join the corresponding feedback table.
  // This join selects the latest metric feedback (using ROW_NUMBER window function)
  // and applies a condition based on the metric threshold.
  // -------------------------------------------------------------------
  if (metric_filter) {
    // Choose the correct feedback table (BooleanMetricFeedback or FloatMetricFeedback).
    const feedback_table = getFeedbackTable(metric_filter.metric_type);
    // Build the condition for filtering based on the metric threshold.
    const reward_condition = `AND value ${metric_filter.operator} {metric_threshold:Float}`;
    // Append the JOIN clause for the metric feedback.
    query += ` JOIN (
      SELECT
        target_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
      FROM ${feedback_table}
      WHERE metric_name = {metric_name:String}
      ${reward_condition}
    ) AS feedback ON ${tableName}.${metric_filter.join_on} = feedback.target_id AND feedback.rn = 1`;
    // Set the query parameters for metric filtering.
    queryParams.metric_name = metric_filter.metric;
    queryParams.metric_threshold = metric_filter.threshold;
  }

  // -------------------------------------------------------------------
  // Demonstration Join Logic:
  // If join_demonstrations is true, join the DemonstrationFeedback table.
  // This join selects the latest demonstration feedback and uses its value as the output.
  // -------------------------------------------------------------------
  if (output_source === "demonstration") {
    query += ` JOIN (
      SELECT
        inference_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
      FROM DemonstrationFeedback
    ) AS demo ON ${tableName}.id = demo.inference_id AND demo.rn = 1`;
  }

  // Append any extra WHERE clauses provided by the caller.
  if (extra_where && extra_where.length > 0) {
    whereClauses.push(...extra_where);
  }

  // If any WHERE conditions have been added, append them to the query.
  if (whereClauses.length > 0) {
    query += " WHERE " + whereClauses.join(" AND ");
  }

  // Append LIMIT and OFFSET clauses if provided.
  if (limit !== undefined) {
    query += " LIMIT {limit:UInt32}";
    queryParams.limit = limit;
  }
  if (offset !== undefined) {
    query += " OFFSET {offset:UInt32}";
    queryParams.offset = offset;
  }

  return { query, query_params: queryParams };
}

/**
 * Executes the constructed query to select rows from the dataset.
 *
 * @param params - The query parameters.
 * @returns A promise resolving to an array of dataset rows matching the query.
 */
export async function selectRowsForDataset(
  params: DatasetQueryParams,
): Promise<DatapointInsert[]> {
  const { query, query_params } = buildDatasetSelectQuery(params);
  const resultSet = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params,
  });
  const rows = await resultSet.json<DatapointInsert[]>();
  return z.array(DatapointInsertSchema).parse(rows);
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
  // Validate that no limit or offset is provided.
  if (params.limit !== undefined || params.offset !== undefined) {
    throw new Error(
      "limit and offset are not supported for countRowsForDataset",
    );
  }

  const { query, query_params } = buildDatasetSelectQuery(params);
  const count_query = `SELECT toUInt32(count()) as count FROM (${query})`;
  const resultSet = await getClickhouseClient().query({
    query: count_query,
    format: "JSONEachRow",
    query_params,
  });
  const rows = await resultSet.json<{ count: number }>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}

/**
 * Helper function to get the correct feedback table based on metric type.
 *
 * @param metric_type - The metric type ("boolean" or "float").
 * @returns The name of the feedback table to use.
 */
function getFeedbackTable(metric_type: "boolean" | "float") {
  return metric_type === "boolean"
    ? "BooleanMetricFeedback"
    : "FloatMetricFeedback";
}

/*
Get name and count for all datasets.
This function should sum the counts of chat and json inferences for each dataset.
The groups should be ordered by last_updated in descending order.
*/
export async function getDatasetCounts({
  function_name,
  page_size,
  offset,
}: {
  function_name?: string;
  page_size?: number;
  offset?: number;
}): Promise<DatasetCountInfo[]> {
  const functionWhereClause = function_name
    ? `AND function_name = {function_name:String}`
    : "";
  const resultSet = await getClickhouseClient().query({
    query: `
      SELECT
        dataset_name,
        toUInt32(sum(count)) AS count,
        formatDateTime(max(last_updated), '%Y-%m-%dT%H:%i:%SZ') AS last_updated
      FROM (
        SELECT
          dataset_name,
          toUInt32(count()) AS count,
          max(updated_at) AS last_updated
        FROM ChatInferenceDatapoint
        FINAL
        WHERE staled_at IS NULL
        ${functionWhereClause}
        GROUP BY dataset_name
        UNION ALL
        SELECT
          dataset_name,
          toUInt32(count()) AS count,
          max(updated_at) AS last_updated
        FROM JsonInferenceDatapoint
        FINAL
        WHERE staled_at IS NULL
        ${functionWhereClause}
        GROUP BY dataset_name
      )
      GROUP BY dataset_name
      ORDER BY last_updated DESC
      ${page_size ? "LIMIT {page_size:UInt32}" : ""}
      ${offset ? "OFFSET {offset:UInt32}" : ""}
    `,
    format: "JSONEachRow",
    query_params: {
      page_size,
      offset,
      function_name: function_name || null,
    },
  });
  const rows = await resultSet.json<DatasetCountInfo[]>();
  return z.array(DatasetCountInfoSchema).parse(rows);
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

/**
 * Executes an INSERT INTO ... SELECT ... query to insert rows into the dataset table.
 *
 * The destination table is determined by the provided `inferenceType`:
 * - "chat"  → ChatInferenceDatapoint
 * - "json"  → JsonInferenceDatapoint
 *
 * This function wraps the query generated by buildDatasetSelectQuery in a subquery
 * to prepend a constant `dataset_name` column.
 * If there is a datapoint with the same `source_inference_id` and `function_name`
 * in the destination table, it will be skipped.
 *
 * @param params - The dataset query parameters.
 * @returns The number of rows inserted.
 */
export async function insertRowsForDataset(
  params: DatasetQueryParams,
): Promise<number> {
  // Validate input parameters
  const validatedParams = DatasetQueryParamsSchema.safeParse(params);
  if (!validatedParams.success) {
    throw new Error(
      `Invalid dataset query params: ${validatedParams.error.message}`,
    );
  }
  if (!validatedParams.data.dataset_name) {
    throw new Error("dataset_name is required for dataset insertion");
  }
  validateDatasetName(validatedParams.data.dataset_name);

  // Determine the destination table based on the inference type
  const destinationTable =
    validatedParams.data.inferenceType === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";

  // Build the SELECT query from the source table
  const { query: sourceQuery, query_params } = buildDatasetSelectQuery(params);
  query_params.datapoint_table = destinationTable;
  query_params.dataset_name = validatedParams.data.dataset_name;

  // Wrap the select query to include all required columns with their defaults
  const wrappedQuery = `
    INSERT INTO {datapoint_table:Identifier}
    SELECT
      {dataset_name:String} as dataset_name,
      subquery.function_name as function_name,
      generateUUIDv7() as id,
      subquery.episode_id as episode_id,
      subquery.input as input,
      subquery.output as output,
      ${validatedParams.data.inferenceType === "chat" ? "subquery.tool_params" : "subquery.output_schema"},
      subquery.tags as tags,
      subquery.auxiliary as auxiliary,
      false as is_deleted,
      now64() as updated_at,
      null as staled_at,
      subquery.id as source_inference_id,
      false as is_custom, -- if we are using the dataset builder implemented here, the datapoints are not custom,
      subquery.name as name
    FROM (
      ${sourceQuery}
    ) AS subquery
    LEFT JOIN {datapoint_table:Identifier} as existing FINAL
      ON {dataset_name:String} = existing.dataset_name
         AND subquery.function_name = existing.function_name
         AND subquery.id = existing.source_inference_id
         AND existing.staled_at IS NULL
      WHERE existing.source_inference_id IS NULL
    `;

  // Execute the INSERT query
  const resultSet = await getClickhouseClient().query({
    query: wrappedQuery,
    query_params,
  });
  const responseHeaders = resultSet.response_headers;
  const summary = responseHeaders["x-clickhouse-summary"] as string;
  const parsedSummary = JSON.parse(summary);
  // NOTE: it seems like recent versions of clickhouse (later than 24.12)
  // don't return the written_rows if it is 0 so we handle that case here
  const writtenRows = Number(parsedSummary.written_rows) || 0;
  return writtenRows;
}

export async function getDatasetRows(
  dataset_name: string,
  page_size: number,
  offset: number,
): Promise<DatasetDetailRow[]> {
  // Ensure offset is not negative
  const validOffset = Math.max(0, offset);

  const query = `
      SELECT *
      FROM (
        SELECT
          id,
          'chat' as type,
          function_name,
          name,
          episode_id,
          formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
        FROM ChatInferenceDatapoint
        FINAL
        WHERE dataset_name = {dataset_name:String} AND staled_at IS NULL
        UNION ALL
        SELECT
          id,
          'json' as type,
          function_name,
          name,
          episode_id,
          formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
        FROM JsonInferenceDatapoint
        FINAL
        WHERE dataset_name = {dataset_name:String} AND staled_at IS NULL
      )
      ORDER BY updated_at DESC, id DESC
      LIMIT {page_size:UInt32}
      OFFSET {offset:UInt32}
    `;

  const resultSet = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      dataset_name,
      page_size,
      offset: validOffset,
    },
  });
  const rows = await resultSet.json<DatasetDetailRow[]>();
  return z.array(DatasetDetailRowSchema).parse(rows);
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
  const parsedRows = rows.map((row) => ({
    previous_id: row.previous_id ?? null,
    next_id: row.next_id ?? null,
  }));
  return parsedRows[0];
}
