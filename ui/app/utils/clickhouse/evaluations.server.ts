import { getConfig } from "../config/index.server";
import { clickhouseClient } from "./client.server";
import {
  EvaluationRunInfoSchema,
  EvaluationStatisticsSchema,
  type EvaluationResult,
  type EvaluationRunInfo,
  type EvaluationStatistics,
  parseEvaluationResult,
  type EvalInfoResult,
  evalInfoResultSchema,
  getEvaluatorMetricName,
  type EvaluationResultWithVariant,
  parseEvaluationResultWithVariant,
  type ParsedEvaluationResultWithVariant,
} from "./evaluations";
import { getStaledWindowQuery, uuidv7ToTimestamp } from "./helpers";

export async function getEvalRunInfos(
  eval_run_ids: string[],
  function_name: string,
): Promise<EvaluationRunInfo[]> {
  const query = `
    SELECT DISTINCT run_tag.value as eval_run_id, run_tag.variant_name as variant_name
    FROM TagInference AS run_tag
    WHERE run_tag.key = 'tensorzero::eval_run_id'
      AND run_tag.value IN ({eval_run_ids:Array(String)})
      AND run_tag.function_name = {function_name:String}
    ORDER BY toUInt128(toUUID(eval_run_id)) DESC
  `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      eval_run_ids: eval_run_ids,
      function_name: function_name,
    },
  });

  const rows = await result.json<EvaluationRunInfo[]>();
  return rows.map((row) => EvaluationRunInfoSchema.parse(row));
}

export async function getEvalResults(
  dataset_name: string,
  function_name: string,
  function_type: "chat" | "json",
  metric_names: string[],
  eval_run_ids: string[],
  limit: number = 100,
  offset: number = 0,
) {
  const datapoint_table_name =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";
  const eval_run_timestamps = eval_run_ids.map((id) => uuidv7ToTimestamp(id));
  const inference_table_name =
    function_type === "chat" ? "ChatInference" : "JsonInference";
  const staled_window_query = getStaledWindowQuery(eval_run_timestamps);
  const query = `
  SELECT
    dp.input as input,
    dp.id as datapoint_id,
    dp.output as reference_output,
    ci.output as generated_output,
    ci.tags['tensorzero::eval_run_id'] as eval_run_id,
    feedback.metric_name as metric_name,
    feedback.value as metric_value
  FROM (
    SELECT DISTINCT dp.id as dp_id
    FROM {datapoint_table_name:Identifier} dp
    INNER JOIN TagInference datapoint_tag
      ON dp.id = toUUIDOrNull(datapoint_tag.value)
      AND datapoint_tag.key = 'tensorzero::datapoint_id'
      AND datapoint_tag.function_name = {function_name:String}
    INNER JOIN {inference_table_name:Identifier} ci
      ON ci.id = datapoint_tag.inference_id
      AND ci.function_name = {function_name:String}
    WHERE dp.dataset_name = {dataset_name:String}
      AND dp.function_name = {function_name:String}
      AND (ci.tags['tensorzero::eval_run_id'] IS NULL OR
           ci.tags['tensorzero::eval_run_id'] IN ({eval_run_ids:Array(String)}))
    ORDER BY toUInt128(toUUID(dp.id)) DESC
    LIMIT {limit:UInt32}
    OFFSET {offset:UInt32}
  ) paginated_dp
  INNER JOIN {datapoint_table_name:Identifier} dp
    ON dp.id = paginated_dp.dp_id
  INNER JOIN TagInference datapoint_tag FINAL
    ON dp.id = toUUIDOrNull(datapoint_tag.value)
    AND datapoint_tag.key = 'tensorzero::datapoint_id'
    AND datapoint_tag.function_name = {function_name:String}
  INNER JOIN {inference_table_name:Identifier} ci
    ON datapoint_tag.inference_id = ci.id
    AND ci.function_name = {function_name:String}
    AND ci.variant_name = datapoint_tag.variant_name
    AND ci.episode_id = datapoint_tag.episode_id
  INNER JOIN (
    SELECT target_id, metric_name, toString(value) as value
    FROM BooleanMetricFeedback
    WHERE metric_name IN ({metric_names:Array(String)})
    UNION ALL
    SELECT target_id, metric_name, toString(value) as value
    FROM FloatMetricFeedback
    WHERE metric_name IN ({metric_names:Array(String)})
  ) feedback
    ON feedback.target_id = ci.id
  WHERE
    (ci.tags['tensorzero::eval_run_id'] IS NULL OR
     ci.tags['tensorzero::eval_run_id'] IN ({eval_run_ids:Array(String)}))
  `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      dataset_name: dataset_name,
      eval_run_ids: eval_run_ids,
      metric_names: metric_names,
      function_name: function_name,
      datapoint_table_name: datapoint_table_name,
      inference_table_name: inference_table_name,
      limit: limit,
      offset: offset,
    },
  });
  const rows = await result.json<EvaluationResult>();
  return rows.map((row) => parseEvaluationResult(row));
}

/*
For each eval run and metric, we want to know:
 - how many datapoints were used
 - what was the mean and stderr of the metric value
*/
export async function getEvalStatistics(
  dataset_name: string,
  function_name: string,
  function_type: "chat" | "json",
  metric_names: string[],
  eval_run_ids: string[],
) {
  const datapoint_table_name =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";
  const eval_run_timestamps = eval_run_ids.map((id) => uuidv7ToTimestamp(id));
  const inference_table_name =
    function_type === "chat" ? "ChatInference" : "JsonInference";
  const staled_window_query = getStaledWindowQuery(eval_run_timestamps);
  const query = `
  SELECT
    ci.tags['tensorzero::eval_run_id'] AS eval_run_id,
    feedback.metric_name AS metric_name,
    toUInt32(count()) AS datapoint_count,
    avg(toFloat64(feedback.value)) AS mean_metric,
    stddevSamp(toFloat64(feedback.value)) / sqrt(count()) AS stderr_metric
  FROM (
    SELECT *
    FROM {datapoint_table_name:Identifier}
    WHERE dataset_name = {dataset_name:String}
      AND function_name = {function_name:String}
      AND (
        ${staled_window_query}
      )
  ) dp
  LEFT JOIN TagInference datapoint_tag FINAL
    ON dp.id = toUUIDOrNull(datapoint_tag.value)
    AND datapoint_tag.key = 'tensorzero::datapoint_id'
    AND datapoint_tag.function_name = {function_name:String}
  LEFT JOIN {inference_table_name:Identifier} ci
    ON datapoint_tag.inference_id = ci.id
    AND ci.function_name = {function_name:String}
    AND ci.variant_name = datapoint_tag.variant_name
    AND ci.episode_id = datapoint_tag.episode_id
  LEFT JOIN (
    SELECT target_id, metric_name, value
    FROM BooleanMetricFeedback
    WHERE metric_name IN ({metric_names:Array(String)})
    UNION ALL
    SELECT target_id, metric_name, value
    FROM FloatMetricFeedback
    WHERE metric_name IN ({metric_names:Array(String)})
  ) feedback
    ON feedback.target_id = ci.id
  WHERE
     ci.tags['tensorzero::eval_run_id'] IN ({eval_run_ids:Array(String)})
    AND feedback.value IS NOT NULL
  GROUP BY
    ci.tags['tensorzero::eval_run_id'],
    feedback.metric_name
  `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      dataset_name: dataset_name,
      eval_run_ids: eval_run_ids,
      metric_names: metric_names,
      function_name: function_name,
      datapoint_table_name: datapoint_table_name,
      inference_table_name: inference_table_name,
    },
  });
  const rows = await result.json<EvaluationStatistics>();
  return rows.map((row) => EvaluationStatisticsSchema.parse(row));
}

export async function countDatapointsForEval(
  dataset_name: string,
  function_name: string,
  function_type: "chat" | "json",
  eval_run_ids: string[],
) {
  const datapoint_table_name =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";
  const eval_run_timestamps = eval_run_ids.map((id) => uuidv7ToTimestamp(id));
  const staled_window_query = getStaledWindowQuery(eval_run_timestamps);

  const query = `
      SELECT toUInt32(count()) as count
      FROM {datapoint_table_name:Identifier}
      WHERE dataset_name = {dataset_name:String}
        AND function_name = {function_name:String}
        AND (
          ${staled_window_query}
        )
  `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      dataset_name: dataset_name,
      function_name: function_name,
      datapoint_table_name: datapoint_table_name,
    },
  });
  const rows = await result.json<{ count: number }>();
  return rows[0].count;
}

export async function countTotalEvalRuns() {
  const query = `
    SELECT toUInt32(uniqExact(value)) as count FROM TagInference WHERE key = 'tensorzero::eval_run_id'
  `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
  });
  const rows = await result.json<{ count: number }>();
  return rows[0].count;
}

export async function getEvalRunInfo(limit: number = 100, offset: number = 0) {
  const query = `
    SELECT
    t1.value AS eval_run_id,
    t2.value AS eval_name,
    t1.function_name,
    t1.variant_name,
    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(max(toUInt128(t1.inference_id)))), '%Y-%m-%dT%H:%i:%SZ') AS last_inference_timestamp
FROM tensorzero_ui_fixtures.TagInference t1
JOIN tensorzero_ui_fixtures.TagInference t2
    ON t1.inference_id = t2.inference_id
WHERE t1.key = 'tensorzero::eval_run_id'
  AND t2.key = 'tensorzero::eval_name'
GROUP BY
    t1.value,
    t2.value,
      t1.function_name,
      t1.variant_name
    ORDER BY toUInt128(toUUID(t1.value)) DESC
    LIMIT {limit:UInt32}
    OFFSET {offset:UInt32}
  `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      limit: limit,
      offset: offset,
    },
  });
  const rows = await result.json<EvalInfoResult>();
  return rows.map((row) => evalInfoResultSchema.parse(row));
}

/*
Returns a map of eval run ids to their most recent inference dates.
For each eval run id, returns the maximum of:
1. The timestamp from the eval run id itself (derived from UUIDv7)
2. The timestamp of the most recent inference associated with that eval run id
*/
export async function getMostRecentEvalInferenceDate(
  eval_run_ids: string[],
): Promise<Map<string, Date>> {
  const query = `
  SELECT
    value as eval_run_id,
    formatDateTime(max(UUIDv7ToDateTime(inference_id)), '%Y-%m-%dT%H:%i:%SZ') as last_inference_timestamp
  FROM TagInference
  WHERE key = 'tensorzero::eval_run_id'
    AND value IN ({eval_run_ids:Array(String)})
  GROUP BY value
  `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      eval_run_ids: eval_run_ids,
    },
  });

  const rows = await result.json<{
    eval_run_id: string;
    last_inference_timestamp: string;
  }>();

  // Create a map of eval_run_id to its last inference timestamp
  const inferenceTimestampMap = new Map<string, Date>();
  rows.forEach((row) => {
    inferenceTimestampMap.set(
      row.eval_run_id,
      new Date(row.last_inference_timestamp),
    );
  });

  // For each eval_run_id, determine the max of its UUID timestamp and its last inference timestamp
  // This handles the case where the eval run id is newer than the last inference timestamp
  // Only possible if there are no inferences for that eval run id yet (we should still return the eval run id timestamp)
  const resultMap = new Map<string, Date>();
  eval_run_ids.forEach((id) => {
    const uuidTimestamp = uuidv7ToTimestamp(id);
    const inferenceTimestamp =
      inferenceTimestampMap.get(id) || new Date("1970-01-01T00:00:00Z");
    resultMap.set(
      id,
      uuidTimestamp > inferenceTimestamp ? uuidTimestamp : inferenceTimestamp,
    );
  });

  return resultMap;
}

export async function searchEvalRuns(
  eval_name: string,
  function_name: string,
  search_query: string,
  limit: number = 100,
  offset: number = 0,
) {
  // This query is not efficient since it is joining on the inference id.
  // We should rewrite this with some kind of MV for performance reasons if it becomes a bottleneck.
  const query = `
    SELECT DISTINCT run_tag.value as eval_run_id, run_tag.variant_name as variant_name
    FROM TagInference AS name_tag
    INNER JOIN TagInference AS run_tag
      ON name_tag.inference_id = run_tag.inference_id
    WHERE name_tag.key = 'tensorzero::eval_name'
      AND name_tag.value = {eval_name:String}
      AND run_tag.key = 'tensorzero::eval_run_id'
      AND run_tag.function_name = {function_name:String}
      AND (positionCaseInsensitive(run_tag.value, {search_query:String}) > 0 OR positionCaseInsensitive(run_tag.variant_name, {search_query:String}) > 0)
    ORDER BY toUInt128(toUUID(eval_run_id)) DESC
    LIMIT {limit:UInt32}
    OFFSET {offset:UInt32}
    `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      eval_name: eval_name,
      function_name: function_name,
      limit: limit,
      offset: offset,
      search_query: search_query,
    },
  });
  const rows = await result.json<EvaluationRunInfo[]>();
  return rows.map((row) => EvaluationRunInfoSchema.parse(row));
}

export async function getEvalsForDatapoint(
  eval_name: string,
  datapoint_id: string,
  eval_run_ids: string[],
): Promise<ParsedEvaluationResultWithVariant[]> {
  const config = await getConfig();
  const function_name = config.evals[eval_name].function_name;
  const dataset_name = config.evals[eval_name].dataset_name;
  if (!function_name) {
    throw new Error(`Eval ${eval_name} not found in config`);
  }
  const function_config = config.functions[function_name];
  const function_type = function_config.type;
  const inference_table_name =
    function_type === "chat" ? "ChatInference" : "JsonInference";
  const datapoint_table_name =
    function_type === "chat"
      ? "ChatInferenceDatapoint"
      : "JsonInferenceDatapoint";

  const evaluators = config.evals[eval_name].evaluators;
  const metric_names = Object.keys(evaluators).map((evaluatorName) =>
    getEvaluatorMetricName(eval_name, evaluatorName),
  );

  const query = `
  SELECT
    inference.input as input,
    dp.id as datapoint_id,
    dp.output as reference_output,
    inference.output as generated_output,
    inference.tags['tensorzero::eval_run_id'] as eval_run_id,
    inference.variant_name as variant_name,
    feedback.metric_name as metric_name,
    feedback.value as metric_value
  FROM TagInference datapoint_tag FINAL
  JOIN {inference_table_name:Identifier} inference
    ON datapoint_tag.inference_id = inference.id
    AND inference.function_name = {function_name:String}
    AND inference.variant_name = datapoint_tag.variant_name
    AND inference.episode_id = datapoint_tag.episode_id
  JOIN {datapoint_table_name:Identifier} dp
    ON dp.id = toUUIDOrNull(datapoint_tag.value)
    AND dp.function_name = {function_name:String}
    AND dp.dataset_name = {dataset_name:String}
  LEFT JOIN (
    SELECT target_id, metric_name, toString(value) as value
    FROM BooleanMetricFeedback
    WHERE metric_name IN ({metric_names:Array(String)})
    UNION ALL
    SELECT target_id, metric_name, toString(value) as value
    FROM FloatMetricFeedback
    WHERE metric_name IN ({metric_names:Array(String)})
  ) feedback
    ON feedback.target_id = inference.id
  WHERE datapoint_tag.key = 'tensorzero::datapoint_id'
    AND datapoint_tag.value = {datapoint_id:String}
    AND inference.tags['tensorzero::eval_run_id'] IN ({eval_run_ids:Array(String)})
  `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      function_name,
      metric_names,
      datapoint_id,
      eval_run_ids,
      inference_table_name,
      datapoint_table_name,
      dataset_name,
    },
  });
  const rows = await result.json<EvaluationResultWithVariant>();
  const parsed_rows = rows.map((row) => parseEvaluationResultWithVariant(row));
  return parsed_rows;
}
