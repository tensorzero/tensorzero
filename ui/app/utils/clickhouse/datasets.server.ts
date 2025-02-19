import z from "zod";
import { clickhouseClient } from "./client.server";
import {
  DatasetCountInfoSchema,
  DatasetInsertSchema,
  DatasetQueryParamsSchema,
  type DatasetCountInfo,
  type DatasetInsert,
  type DatasetQueryParams,
} from "./datasets";

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
): Promise<DatasetInsert[]> {
  const { query, query_params } = buildDatasetSelectQuery(params);
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params,
  });
  const rows = await resultSet.json<DatasetInsert[]>();
  return z.array(DatasetInsertSchema).parse(rows);
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
  const resultSet = await clickhouseClient.query({
    query: count_query,
    format: "JSONEachRow",
    query_params,
  });
  const rows = await resultSet.json<{ count: number }>();
  return rows[0].count;
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
export async function getDatasetCounts(): Promise<DatasetCountInfo[]> {
  const resultSet = await clickhouseClient.query({
    query: `
      SELECT
        dataset_name,
        toUInt32(sum(count)) AS count,
        formatDateTime(max(last_updated), '%Y-%m-%dT%H:%i:%SZ') AS last_updated
      FROM (
        SELECT
          dataset_name,
          toUInt32(count()) AS count,
          max(created_at) AS last_updated
        FROM ChatInferenceDataset
        WHERE is_deleted = false
        GROUP BY dataset_name
        UNION ALL
        SELECT
          dataset_name,
          toUInt32(count()) AS count,
          max(created_at) AS last_updated
        FROM JsonInferenceDataset
        WHERE is_deleted = false
        GROUP BY dataset_name
      )
      GROUP BY dataset_name
      ORDER BY last_updated DESC
    `,
    format: "JSONEachRow",
  });
  const rows = await resultSet.json<DatasetCountInfo[]>();
  return z.array(DatasetCountInfoSchema).parse(rows);
}
/**
 * Executes an INSERT INTO ... SELECT ... query to insert rows into the dataset table.
 *
 * The destination table is determined by the provided `inferenceType`:
 * - "chat"  → ChatInferenceDataset
 * - "json"  → JsonInferenceDataset
 *
 * This function wraps the query generated by buildDatasetSelectQuery in a subquery
 * to prepend a constant `dataset_name` column.
 *
 * @param params - The dataset query parameters.
 * @returns A promise that resolves when the insert query completes.
 */
export async function insertRowsForDataset(
  params: DatasetQueryParams,
): Promise<void> {
  const validatedParams = DatasetQueryParamsSchema.safeParse(params);
  if (!validatedParams.success) {
    throw new Error(
      `Invalid dataset query params: ${validatedParams.error.message}`,
    );
  }
  if (!validatedParams.data.dataset_name) {
    throw new Error("dataset_name is required for dataset insertion");
  }

  const destinationTable =
    validatedParams.data.inferenceType === "chat"
      ? "ChatInferenceDataset"
      : "JsonInferenceDataset";

  // Build the SELECT query from the source table
  const { query: sourceQuery, query_params } = buildDatasetSelectQuery(params);

  // Wrap the select query to include all required columns with their defaults
  const wrappedQuery = `
    INSERT INTO ${destinationTable}
    SELECT
      '${validatedParams.data.dataset_name}' as dataset_name,
      function_name,
      id,
      episode_id,
      input,
      output,
      ${validatedParams.data.inferenceType === "chat" ? "tool_params" : "output_schema"},
      tags,
      auxiliary,
      false as is_deleted,
      now() as updated_at
    FROM (
      ${sourceQuery}
    ) AS t
  `;

  // Execute the INSERT query
  await clickhouseClient.query({
    query: wrappedQuery,
    query_params,
  });
}
