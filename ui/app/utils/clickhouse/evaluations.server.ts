import { clickhouseClient } from "./client.server";
import {
  EvaluationResultSchema,
  EvaluationRunInfoSchema,
  EvaluationStatisticsSchema,
  type EvaluationResult,
  type EvaluationRunInfo,
  type EvaluationStatistics,
} from "./evaluations";
import { getStaledWindowQuery, uuidv7ToTimestamp } from "./helpers";

export async function getEvalRunIds(
  eval_name: string,
  limit: number = 100,
  offset: number = 0,
): Promise<EvaluationRunInfo[]> {
  const query = `
    SELECT DISTINCT run_tag.value as eval_run_id, run_tag.variant_name as variant_name
    FROM TagInference AS name_tag
    INNER JOIN TagInference AS run_tag
      ON name_tag.inference_id = run_tag.inference_id
    WHERE name_tag.key = 'tensorzero::eval_name'
      AND name_tag.value = {eval_name:String}
      AND run_tag.key = 'tensorzero::eval_run_id'
    ORDER BY toUInt128(toUUID(eval_run_id)) DESC
    LIMIT {limit:UInt32}
    OFFSET {offset:UInt32}
    `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      eval_name: eval_name,
      limit: limit,
      offset: offset,
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
    SELECT *
    FROM {datapoint_table_name:Identifier}
    WHERE dataset_name = {dataset_name:String}
      AND function_name = {function_name:String}
      AND (
        ${staled_window_query}
      )
    ORDER BY toUInt128(id) DESC
    LIMIT {limit:UInt32}
    OFFSET {offset:UInt32}
  ) dp
  LEFT JOIN TagInference datapoint_tag
    ON dp.id = toUUIDOrNull(datapoint_tag.value)
    AND datapoint_tag.key = 'tensorzero::datapoint_id'
    AND datapoint_tag.function_name = {function_name:String}
  LEFT JOIN {inference_table_name:Identifier} ci
    ON datapoint_tag.inference_id = ci.id
    AND ci.function_name = {function_name:String}
    AND ci.variant_name = datapoint_tag.variant_name
    AND ci.episode_id = datapoint_tag.episode_id
  LEFT JOIN (
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
  return rows.map((row) => EvaluationResultSchema.parse(row));
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
  LEFT JOIN TagInference datapoint_tag
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
    (ci.tags['tensorzero::eval_run_id'] IS NULL OR
     ci.tags['tensorzero::eval_run_id'] IN ({eval_run_ids:Array(String)}))
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

export function getEvaluatorMetricName(
  evalName: string,
  evaluatorName: string,
): string {
  return `tensorzero::eval_name::${evalName}::evaluator_name::${evaluatorName}`;
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
