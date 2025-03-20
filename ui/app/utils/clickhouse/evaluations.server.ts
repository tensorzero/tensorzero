import { clickhouseClient } from "./client.server";
import {
  EvaluationResultSchema,
  EvaluationRunInfoSchema,
  type EvaluationResult,
  type EvaluationRunInfo,
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
    ORDER BY eval_run_id, variant_name
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
      ORDER BY id
      LIMIT {limit:UInt32}
      OFFSET {offset:UInt32}
    ) dp
    INNER JOIN TagInference datapoint_tag
      ON dp.id = toUUID(datapoint_tag.value)
      AND datapoint_tag.key = 'tensorzero::datapoint_id'
      AND datapoint_tag.function_name = {function_name:String}
    INNER JOIN {inference_table_name:Identifier} ci
      ON datapoint_tag.inference_id = ci.id
      AND ci.function_name = {function_name:String}
      AND ci.variant_name = datapoint_tag.variant_name
      AND ci.episode_id = datapoint_tag.episode_id
    INNER JOIN (
      SELECT target_id, metric_name, toString(value) as value FROM BooleanMetricFeedback
      WHERE metric_name IN ({metric_names:Array(String)})
      UNION ALL
      SELECT target_id, metric_name, toString(value) as value FROM FloatMetricFeedback
      WHERE metric_name IN ({metric_names:Array(String)})
    ) feedback
      ON feedback.target_id = ci.id
    WHERE
      ci.tags['tensorzero::eval_run_id'] IN ({eval_run_ids:Array(String)})
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
