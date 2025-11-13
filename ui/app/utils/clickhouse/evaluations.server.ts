import { logger } from "~/utils/logger";
import { getConfig, getFunctionConfig } from "../config/index.server";
import { resolveInput } from "../resolve.server";
import { getClickhouseClient } from "./client.server";
import { CountSchema, inputSchema } from "./common";
import {
  EvaluationRunInfoSchema,
  EvaluationStatisticsSchema,
  type EvaluationResult,
  type EvaluationRunInfo,
  type EvaluationStatistics,
  type EvaluationInfoResult,
  evaluationInfoResultSchema,
  getEvaluatorMetricName,
  type EvaluationResultWithVariant,
  type ParsedEvaluationResultWithVariant,
  type ParsedEvaluationResult,
  JsonEvaluationResultSchema,
  ChatEvaluationResultSchema,
  ParsedEvaluationResultWithVariantSchema,
  type EvaluationRunSearchResult,
  EvaluationRunSearchResultSchema,
} from "./evaluations";

export async function getEvaluationRunInfos(
  evaluation_run_ids: string[],
  function_name: string,
): Promise<EvaluationRunInfo[]> {
  const query = `
    SELECT
      any(run_tag.value) as evaluation_run_id,
      any(run_tag.variant_name) as variant_name,
      formatDateTime(
        max(UUIDv7ToDateTime(inference_id)),
        '%Y-%m-%dT%H:%i:%SZ'
      ) as most_recent_inference_date
    FROM
      TagInference AS run_tag FINAL
    WHERE
      run_tag.key = 'tensorzero::evaluation_run_id'
      AND run_tag.value IN ({evaluation_run_ids:Array(String)})
      AND run_tag.function_name = {function_name:String}
    GROUP BY
      run_tag.value
    ORDER BY
      toUInt128(toUUID(evaluation_run_id)) DESC
  `;

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      evaluation_run_ids: evaluation_run_ids,
      function_name: function_name,
    },
  });

  const rows = await result.json<EvaluationRunInfo[]>();
  return rows.map((row) => EvaluationRunInfoSchema.parse(row));
}

export async function getEvaluationRunInfosForDatapoint(
  datapoint_id: string,
  function_name: string,
): Promise<EvaluationRunInfo[]> {
  const config = await getConfig();
  const functionConfig = await getFunctionConfig(function_name, config);
  const function_type = functionConfig?.type;
  if (!function_type) {
    throw new Error(`Function ${function_name} not found in config`);
  }
  const inference_table_name =
    function_type === "chat" ? "ChatInference" : "JsonInference";

  const query = `
    WITH datapoint_inference_ids AS (
      SELECT inference_id FROM TagInference FINAL WHERE key = 'tensorzero::datapoint_id' AND value = {datapoint_id:String}
    )
    SELECT any(tags['tensorzero::evaluation_run_id']) as evaluation_run_id,
           any(variant_name) as variant_name,
           formatDateTime(
             max(UUIDv7ToDateTime(id)),
             '%Y-%m-%dT%H:%i:%SZ'
           ) as most_recent_inference_date
    FROM {inference_table_name:Identifier}
      WHERE id IN (SELECT inference_id FROM datapoint_inference_ids)
      AND function_name = {function_name:String}
    GROUP BY
      tags['tensorzero::evaluation_run_id']
  `;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      datapoint_id: datapoint_id,
      function_name: function_name,
      inference_table_name: inference_table_name,
    },
  });
  const rows = await result.json<EvaluationRunInfo[]>();
  return rows.map((row) => EvaluationRunInfoSchema.parse(row));
}

async function parseEvaluationResult(
  result: EvaluationResult,
  function_name: string,
): Promise<ParsedEvaluationResult> {
  // Parse the input field
  const parsedInput = inputSchema.parse(JSON.parse(result.input));
  const config = await getConfig();
  const functionConfig = await getFunctionConfig(function_name, config);
  const resolvedInput = await resolveInput(parsedInput, functionConfig);

  // Parse the outputs
  const generatedOutput = JSON.parse(result.generated_output);
  const referenceOutput = JSON.parse(result.reference_output);
  // Determine if this is a chat result by checking if generated_output is an array
  if (Array.isArray(generatedOutput)) {
    // This is likely a chat evaluation result
    return ChatEvaluationResultSchema.parse({
      ...result,
      input: resolvedInput,
      generated_output: generatedOutput,
      reference_output: referenceOutput,
    });
  } else {
    // This is likely a JSON evaluation result
    return JsonEvaluationResultSchema.parse({
      ...result,
      input: resolvedInput,
      generated_output: generatedOutput,
      reference_output: referenceOutput,
    });
  }
}

async function parseEvaluationResultWithVariant(
  result: EvaluationResultWithVariant,
  function_name: string,
): Promise<ParsedEvaluationResultWithVariant> {
  try {
    // Parse using the same logic as parseEvaluationResult
    const parsedResult = await parseEvaluationResult(result, function_name);

    // Add the variant_name to the parsed result
    const parsedResultWithVariant = {
      ...parsedResult,
      variant_name: result.variant_name,
    };
    return ParsedEvaluationResultWithVariantSchema.parse(
      parsedResultWithVariant,
    );
  } catch (error) {
    logger.warn(
      "Failed to parse evaluation result with variant using structure-based detection:",
      error,
    );
    // Fallback to direct parsing if needed
    return ParsedEvaluationResultWithVariantSchema.parse({
      ...result,
      input: result.input,
      generated_output: result.generated_output,
      reference_output: result.reference_output,
    });
  }
}

const getEvaluationResultDatapointIdQuery = (
  limit?: number,
  offset?: number,
) => {
  return `
    all_inference_ids AS (
      SELECT DISTINCT inference_id
      FROM TagInference FINAL WHERE key = 'tensorzero::evaluation_run_id'
      AND function_name = {function_name:String}
      AND value IN ({evaluation_run_ids:Array(String)})
    ),
    all_datapoint_ids AS (
      SELECT DISTINCT value as datapoint_id
      FROM TagInference FINAL
      WHERE key = 'tensorzero::datapoint_id'
      AND function_name = {function_name:String}
      AND inference_id IN (SELECT inference_id FROM all_inference_ids)
      ORDER BY toUInt128(toUUID(datapoint_id)) DESC
      ${limit ? `LIMIT ${limit}` : ""}
      ${offset ? `OFFSET ${offset}` : ""}
    )
`;
};

export async function getEvaluationResults(
  function_name: string,
  function_type: "chat" | "json",
  metric_names: string[],
  evaluation_run_ids: string[],
  limit: number = 100,
  offset: number = 0,
) {
  const datapoint_table_name =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";
  const inference_table_name =
    function_type === "chat" ? "ChatInference" : "JsonInference";
  const query = `
  WITH ${getEvaluationResultDatapointIdQuery(limit, offset)},
    filtered_dp AS (
      SELECT * FROM {datapoint_table_name:Identifier} FINAL
      WHERE function_name = {function_name:String}
      -- We don't have the dataset_name here but there is an index on datapoint_ids so this should not be a full scan
      AND id IN (SELECT datapoint_id FROM all_datapoint_ids)
    ),
    filtered_inference AS (
      SELECT * FROM {inference_table_name:Identifier}
      WHERE id IN (SELECT inference_id FROM all_inference_ids)
      AND function_name = {function_name:String}
    ),
    filtered_feedback AS (
      SELECT metric_name,
             argMax(toString(value), timestamp) as value,
             argMax(tags['tensorzero::evaluator_inference_id'], timestamp) as evaluator_inference_id,
             argMax(id, timestamp) as feedback_id,
             argMax(tags['tensorzero::human_feedback'], timestamp) == 'true' as is_human_feedback,
             target_id
      FROM BooleanMetricFeedback
      WHERE metric_name IN ({metric_names:Array(String)})
      AND target_id IN (SELECT inference_id FROM all_inference_ids)
      GROUP BY target_id, metric_name -- for the argMax
      UNION ALL
      SELECT metric_name,
             argMax(toString(value), timestamp) as value,
             argMax(tags['tensorzero::evaluator_inference_id'], timestamp) as evaluator_inference_id,
             argMax(id, timestamp) as feedback_id,
             argMax(tags['tensorzero::human_feedback'], timestamp) == 'true' as is_human_feedback,
             target_id
      FROM FloatMetricFeedback
      WHERE metric_name IN ({metric_names:Array(String)})
      AND target_id IN (SELECT inference_id FROM all_inference_ids)
      GROUP BY target_id, metric_name -- for the argMax
    )
  SELECT
    dp.input as input,
    dp.id as datapoint_id,
    dp.name as name,
    dp.output as reference_output,
    ci.output as generated_output,
    ci.function_name as function_name,
    ci.tags['tensorzero::evaluation_run_id'] as evaluation_run_id,
    ci.tags['tensorzero::dataset_name'] as dataset_name,
    if(length(feedback.evaluator_inference_id) > 0, feedback.evaluator_inference_id, null) as evaluator_inference_id,
    ci.id as inference_id,
    ci.episode_id as episode_id,
    feedback.metric_name as metric_name,
    feedback.value as metric_value,
    feedback.feedback_id as feedback_id,
    toBool(feedback.is_human_feedback) as is_human_feedback,
    formatDateTime(dp.staled_at, '%Y-%m-%dT%H:%i:%SZ') as staled_at
  FROM filtered_dp dp
  INNER JOIN filtered_inference ci
    ON toUUIDOrNull(ci.tags['tensorzero::datapoint_id']) = dp.id
  LEFT JOIN filtered_feedback feedback
    ON feedback.target_id = ci.id
  ORDER BY toUInt128(datapoint_id) DESC
  `;

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      evaluation_run_ids: evaluation_run_ids,
      metric_names: metric_names,
      function_name: function_name,
      datapoint_table_name: datapoint_table_name,
      inference_table_name: inference_table_name,
      limit: limit,
      offset: offset,
    },
  });
  const rows = await result.json<EvaluationResult>();
  return Promise.all(
    rows.map((row) => parseEvaluationResult(row, function_name)),
  );
}

/*
For each evaluation run and metric, we want to know:
 - how many datapoints were used
 - what was the mean and stderr of the metric value
*/
export async function getEvaluationStatistics(
  function_name: string,
  function_type: "chat" | "json",
  metric_names: string[],
  evaluation_run_ids: string[],
) {
  const datapoint_table_name =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";
  const inference_table_name =
    function_type === "chat" ? "ChatInference" : "JsonInference";
  const query = `
  WITH ${getEvaluationResultDatapointIdQuery()},
    filtered_inference AS (
      SELECT * FROM {inference_table_name:Identifier}
      WHERE id IN (SELECT inference_id FROM all_inference_ids)
      AND function_name = {function_name:String}
    ),
   filtered_feedback AS (
      SELECT metric_name,
             argMax(value, timestamp) as value,
             argMax(tags['tensorzero::evaluator_inference_id'], timestamp) as evaluator_inference_id,
             argMax(id, timestamp) as feedback_id,
             target_id
      FROM BooleanMetricFeedback
      WHERE metric_name IN ({metric_names:Array(String)})
      AND target_id IN (SELECT inference_id FROM all_inference_ids)
      GROUP BY target_id, metric_name -- for the argMax
      UNION ALL
      SELECT metric_name,
             argMax(value, timestamp) as value,
             argMax(tags['tensorzero::evaluator_inference_id'], timestamp) as evaluator_inference_id,
             argMax(id, timestamp) as feedback_id,
             target_id
      FROM FloatMetricFeedback
      WHERE metric_name IN ({metric_names:Array(String)})
      AND target_id IN (SELECT inference_id FROM all_inference_ids)
      GROUP BY target_id, metric_name -- for the argMax
    )
  SELECT
    filtered_inference.tags['tensorzero::evaluation_run_id'] AS evaluation_run_id,
    filtered_feedback.metric_name AS metric_name,
    toUInt32(count()) AS datapoint_count,
    avg(toFloat64(filtered_feedback.value)) AS mean_metric,
    stddevSamp(toFloat64(filtered_feedback.value)) / sqrt(count()) AS stderr_metric
  FROM filtered_inference
  INNER JOIN filtered_feedback
    ON filtered_feedback.target_id = filtered_inference.id
    AND filtered_feedback.value IS NOT NULL
  GROUP BY
    filtered_inference.tags['tensorzero::evaluation_run_id'],
    filtered_feedback.metric_name
  ORDER BY
    toUInt128(toUUID(filtered_inference.tags['tensorzero::evaluation_run_id'])) DESC,
    metric_name ASC
  `;

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      evaluation_run_ids: evaluation_run_ids,
      metric_names: metric_names,
      function_name: function_name,
      datapoint_table_name: datapoint_table_name,
      inference_table_name: inference_table_name,
    },
  });
  const rows = await result.json<EvaluationStatistics>();
  return rows.map((row) => EvaluationStatisticsSchema.parse(row));
}

export async function countDatapointsForEvaluation(
  function_name: string,
  function_type: "chat" | "json",
  evaluation_run_ids: string[],
) {
  const datapoint_table_name =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";
  const inference_table_name =
    function_type === "chat" ? "ChatInference" : "JsonInference";

  const query = `
      WITH ${getEvaluationResultDatapointIdQuery()}
      SELECT toUInt32(count()) as count
      FROM all_datapoint_ids
  `;

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      function_name: function_name,
      datapoint_table_name: datapoint_table_name,
      inference_table_name: inference_table_name,
      evaluation_run_ids: evaluation_run_ids,
    },
  });
  const rows = await result.json<{ count: number }>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}

export async function countTotalEvaluationRuns() {
  const query = `
    SELECT toUInt32(uniqExact(value)) as count FROM TagInference WHERE key = 'tensorzero::evaluation_run_id'
  `;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
  });
  const rows = await result.json<{ count: number }>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}

export async function getEvaluationRunInfo(
  limit: number = 100,
  offset: number = 0,
) {
  const query = `
    SELECT
        evaluation_run_id,
        any(evaluation_name) AS evaluation_name,
        any(inference_function_name) AS function_name,
        any(variant_name) AS variant_name,
        any(dataset_name) AS dataset_name,
        formatDateTime(UUIDv7ToDateTime(tensorzero_uint_to_uuid(max(max_inference_id))), '%Y-%m-%dT%H:%i:%SZ') AS last_inference_timestamp
    FROM (
        SELECT
            maxIf(value, key = 'tensorzero::evaluation_run_id') AS evaluation_run_id,
            maxIf(value, key = 'tensorzero::evaluation_name') AS evaluation_name,
            maxIf(value, key = 'tensorzero::dataset_name') AS dataset_name,
            any(function_name) AS inference_function_name,
            any(variant_name) AS variant_name,
            max(toUInt128(inference_id)) AS max_inference_id
        FROM TagInference FINAL
        WHERE key IN ('tensorzero::evaluation_run_id', 'tensorzero::evaluation_name', 'tensorzero::dataset_name')
        GROUP BY inference_id
    )
    WHERE NOT startsWith(inference_function_name, 'tensorzero::')
    GROUP BY evaluation_run_id
    ORDER BY toUInt128(toUUID(evaluation_run_id)) DESC
    LIMIT {limit:UInt32}
    OFFSET {offset:UInt32}
  `;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      limit: limit,
      offset: offset,
    },
  });
  const rows = await result.json<EvaluationInfoResult>();
  return rows.map((row) => evaluationInfoResultSchema.parse(row));
}

export async function searchEvaluationRuns(
  evaluation_name: string,
  function_name: string,
  search_query: string,
  limit: number = 100,
  offset: number = 0,
) {
  const query = `
    WITH
      evaluation_inference_ids AS (
        SELECT inference_id
        FROM TagInference
        WHERE key = 'tensorzero::evaluation_name'
        AND value = {evaluation_name:String}
      )
    SELECT DISTINCT value as evaluation_run_id, variant_name
    FROM TagInference FINAL
    WHERE key = 'tensorzero::evaluation_run_id'
      AND function_name = {function_name:String}
      AND inference_id IN (SELECT inference_id FROM evaluation_inference_ids)
      AND (positionCaseInsensitive(value, {search_query:String}) > 0 OR positionCaseInsensitive(variant_name, {search_query:String}) > 0)
    ORDER BY toUInt128(toUUID(evaluation_run_id)) DESC
    LIMIT {limit:UInt32}
    OFFSET {offset:UInt32}
    `;

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      evaluation_name: evaluation_name,
      function_name: function_name,
      limit: limit,
      offset: offset,
      search_query: search_query,
    },
  });
  const rows = await result.json<EvaluationRunSearchResult[]>();
  return rows.map((row) => EvaluationRunSearchResultSchema.parse(row));
}

export async function getEvaluationsForDatapoint(
  evaluation_name: string,
  datapoint_id: string,
  evaluation_run_ids: string[],
): Promise<ParsedEvaluationResultWithVariant[]> {
  const config = await getConfig();
  const evaluation_config = config.evaluations[evaluation_name];
  if (!evaluation_config) {
    throw new Error(`Evaluation ${evaluation_name} not found in config`);
  }
  const function_name = evaluation_config.function_name;
  if (!function_name) {
    throw new Error(`evaluation ${evaluation_name} not found in config`);
  }
  const function_config = await getFunctionConfig(function_name, config);
  if (!function_config) {
    throw new Error(`Function ${function_name} not found in config`);
  }
  const function_type = function_config.type;
  const inference_table_name =
    function_type === "chat" ? "ChatInference" : "JsonInference";
  const datapoint_table_name =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";

  const evaluators = evaluation_config.evaluators;
  const metric_names = Object.keys(evaluators).map((evaluatorName) =>
    getEvaluatorMetricName(evaluation_name, evaluatorName),
  );
  const query = `
   WITH all_inference_ids AS (
      SELECT DISTINCT inference_id
      FROM TagInference FINAL WHERE key = 'tensorzero::datapoint_id'
      AND value = {datapoint_id:String}
      AND function_name = {function_name:String}
    ),
    filtered_inference AS (
      SELECT * FROM {inference_table_name:Identifier}
      WHERE id IN (SELECT inference_id FROM all_inference_ids)
      AND function_name = {function_name:String}
      AND tags['tensorzero::evaluation_run_id'] IN ({evaluation_run_ids:Array(String)})
    ),
    filtered_feedback AS (
      SELECT target_id,
            metric_name,
            argMax(toString(value), timestamp) as value,
            argMax(tags['tensorzero::evaluator_inference_id'], timestamp) as evaluator_inference_id,
            argMax(id, timestamp) as feedback_id,
            argMax(tags['tensorzero::human_feedback'], timestamp) == 'true' as is_human_feedback
      FROM BooleanMetricFeedback
      WHERE metric_name IN ({metric_names:Array(String)})
      AND target_id IN (SELECT inference_id FROM all_inference_ids)
      GROUP BY target_id, metric_name -- for the argMax
      UNION ALL
      SELECT target_id,
            metric_name,
            argMax(toString(value), timestamp) as value,
            argMax(tags['tensorzero::evaluator_inference_id'], timestamp) as evaluator_inference_id,
            argMax(id, timestamp) as feedback_id,
            argMax(tags['tensorzero::human_feedback'], timestamp) == 'true' as is_human_feedback
      FROM FloatMetricFeedback
      WHERE metric_name IN ({metric_names:Array(String)})
      AND target_id IN (SELECT inference_id FROM all_inference_ids)
      GROUP BY target_id, metric_name -- for the argMax
    ),
    filtered_datapoint AS (
      SELECT * FROM {datapoint_table_name:Identifier}
      FINAL
      WHERE id = {datapoint_id:UUID}
      AND function_name = {function_name:String}
    )
    SELECT
      filtered_inference.input as input,
      filtered_inference.tags['tensorzero::datapoint_id'] as datapoint_id,
      filtered_datapoint.name as name,
      filtered_datapoint.output as reference_output,
      filtered_inference.id as inference_id,
      filtered_inference.episode_id as episode_id,
      filtered_inference.output as generated_output,
      filtered_inference.tags['tensorzero::evaluation_run_id'] as evaluation_run_id,
      filtered_inference.variant_name as variant_name,
      filtered_inference.tags['tensorzero::dataset_name'] as dataset_name,
      if(length(filtered_feedback.evaluator_inference_id) > 0, filtered_feedback.evaluator_inference_id, null) as evaluator_inference_id,
      filtered_feedback.metric_name as metric_name,
      filtered_feedback.value as metric_value,
      filtered_feedback.feedback_id as feedback_id,
      toBool(filtered_feedback.is_human_feedback) as is_human_feedback,
      formatDateTime(filtered_datapoint.staled_at, '%Y-%m-%dT%H:%i:%SZ') as staled_at
    FROM filtered_inference
    INNER JOIN filtered_datapoint
      ON filtered_datapoint.id = toUUIDOrNull(filtered_inference.tags['tensorzero::datapoint_id'])
    LEFT JOIN filtered_feedback
      ON filtered_feedback.target_id = filtered_inference.id
  `;

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      function_name,
      metric_names,
      datapoint_id,
      evaluation_run_ids,
      inference_table_name,
      datapoint_table_name,
    },
  });
  const rows = await result.json<EvaluationResultWithVariant>();
  const parsed_rows = await Promise.all(
    rows.map((row) => parseEvaluationResultWithVariant(row, function_name)),
  );
  return parsed_rows;
}
/**
 * Polls for evaluations until a specific feedback item is found.
 * @param evaluation_name The name of the evaluation function.
 * @param datapoint_id The ID of the datapoint.
 * @param evaluation_run_ids Array of evaluation run IDs to query.
 * @param new_feedback_id The ID of the feedback item to find.
 * @param max_retries Maximum number of polling attempts.
 * @param retry_delay Delay between retries in milliseconds.
 * @returns An array of parsed evaluation results.
 */
export async function pollForEvaluations(
  evaluation_name: string,
  datapoint_id: string,
  evaluation_run_ids: string[],
  new_feedback_id: string,
  max_retries: number = 10,
  retry_delay: number = 200,
): Promise<ParsedEvaluationResultWithVariant[]> {
  let evaluations: ParsedEvaluationResultWithVariant[] = [];
  let found = false;

  for (let i = 0; i < max_retries; i++) {
    evaluations = await getEvaluationsForDatapoint(
      evaluation_name,
      datapoint_id,
      evaluation_run_ids,
    );

    if (
      evaluations.some(
        (evaluation) => evaluation.feedback_id === new_feedback_id,
      )
    ) {
      found = true;
      break;
    }

    if (i < max_retries - 1) {
      // Don't sleep after the last attempt
      await new Promise((resolve) => setTimeout(resolve, retry_delay));
    }
  }

  if (!found) {
    logger.warn(
      `Evaluation with feedback ${new_feedback_id} for datapoint ${datapoint_id} not found after ${max_retries} retries.`,
    );
  }

  return evaluations;
}

/**
 * Polls for evaluation results until they are available or max retries is reached.
 * This is useful when waiting for newly created evaluations to be available in ClickHouse.
 *
 * @param function_name The name of the function.
 * @param function_type The type of function (chat or json).
 * @param metric_names Array of metric names to query.
 * @param evaluation_run_ids Array of evaluation run IDs to query.
 * @param new_feedback_id The ID of the feedback item to find.
 * @param limit Maximum number of results to return.
 * @param offset Offset for pagination.
 * @param max_retries Maximum number of polling attempts.
 * @param retry_delay Delay between retries in milliseconds.
 * @returns An array of parsed evaluation results.
 */
export async function pollForEvaluationResults(
  function_name: string,
  function_type: "chat" | "json",
  metric_names: string[],
  evaluation_run_ids: string[],
  new_feedback_id: string,
  limit: number = 100,
  offset: number = 0,
  max_retries: number = 10,
  retry_delay: number = 200,
): Promise<ParsedEvaluationResult[]> {
  let results: ParsedEvaluationResult[] = [];
  let found = false;

  for (let i = 0; i < max_retries; i++) {
    results = await getEvaluationResults(
      function_name,
      function_type,
      metric_names,
      evaluation_run_ids,
      limit,
      offset,
    );

    if (results.some((result) => result.feedback_id === new_feedback_id)) {
      found = true;
      break;
    }

    if (i < max_retries - 1) {
      // Don't sleep after the last attempt
      await new Promise((resolve) => setTimeout(resolve, retry_delay));
    }
  }

  if (!found) {
    logger.warn(
      `Evaluation result with feedback ${new_feedback_id} not found after ${max_retries} retries.`,
    );
  }

  return results;
}
