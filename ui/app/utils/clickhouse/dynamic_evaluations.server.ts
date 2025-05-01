import { clickhouseClient } from "./client.server";
import {
  dynamicEvaluationRunEpisodeWithFeedbackSchema,
  dynamicEvaluationRunSchema,
  dynamicEvaluationRunStatisticsByMetricNameSchema,
  type DynamicEvaluationRun,
  type DynamicEvaluationRunEpisodeWithFeedback,
  type DynamicEvaluationRunStatisticsByMetricName,
} from "./dynamic_evaluations";

export async function getDynamicEvaluationRuns(
  page_size: number,
  offset: number,
  run_id?: string,
): Promise<DynamicEvaluationRun[]> {
  const query = `
    SELECT
      run_display_name as name,
      uint_to_uuid(run_id_uint) as id,
      variant_pins,
      tags,
      project_name,
      formatDateTime(UUIDv7ToDateTime(uint_to_uuid(run_id_uint)), '%Y-%m-%dT%H:%i:%SZ') as timestamp
    FROM DynamicEvaluationRun
    ${run_id ? `WHERE toUInt128(toUUID({run_id:String})) = run_id_uint` : ""}
    ORDER BY run_id_uint DESC
    LIMIT {page_size:UInt64} OFFSET {offset:UInt64}
    `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      page_size,
      offset,
      run_id,
    },
  });
  const rows = await result.json<DynamicEvaluationRun[]>();
  return rows.map((row) => dynamicEvaluationRunSchema.parse(row));
}

export async function countDynamicEvaluationRuns(): Promise<number> {
  const query = `
    SELECT toUInt32(count()) as count FROM DynamicEvaluationRun
  `;
  const result = await clickhouseClient.query({ query, format: "JSONEachRow" });
  const rows = await result.json<{ count: number }>();
  return rows[0].count;
}

/**
 * Returns information about the episodes that were used in a dynamic evaluation run,
 * along with the feedback that was collected for each episode.
 *
 * The feedback is given as arrays feedback_metric_names and feedback_values.
 * The arrays are sorted by the metric name.
 */
export async function getDynamicEvaluationRunEpisodesByRunIdWithFeedback(
  page_size: number,
  offset: number,
  run_id: string,
): Promise<DynamicEvaluationRunEpisodeWithFeedback[]> {
  const query = `
    WITH
      episodes AS (
        SELECT
          episode_id_uint,
          run_id_uint,
          tags,
          updated_at,
          datapoint_name as task_name
        FROM DynamicEvaluationRunEpisodeByRunId
        WHERE toUInt128(toUUID({run_id:String})) = run_id_uint
        ORDER BY episode_id_uint DESC
        LIMIT {page_size:UInt64}
        OFFSET {offset:UInt64}
      ),
      feedback_union AS (
        SELECT
          target_id,
          metric_name,
          argMax(toString(value), toUInt128(id)) AS value
        FROM FloatMetricFeedbackByTargetId
        WHERE target_id IN (
          SELECT uint_to_uuid(episode_id_uint) FROM episodes
        )
        GROUP BY target_id, metric_name
        UNION ALL
        SELECT
          target_id,
          metric_name,
          argMax(toString(value), toUInt128(id)) AS value
        FROM BooleanMetricFeedbackByTargetId
        WHERE target_id IN (
          SELECT uint_to_uuid(episode_id_uint) FROM episodes
        )
        GROUP BY target_id, metric_name
        UNION ALL
        SELECT
          target_id,
          'comment' AS metric_name,
          value
        FROM CommentFeedbackByTargetId
        WHERE target_id IN (
          SELECT uint_to_uuid(episode_id_uint) FROM episodes
        )
      )
    SELECT
      uint_to_uuid(e.episode_id_uint) AS episode_id,
      formatDateTime(
        min(e.updated_at), -- when did the episode start?
        '%Y-%m-%dT%H:%i:%SZ'
      ) AS timestamp,
      uint_to_uuid(e.run_id_uint) AS run_id,
      e.tags,
      e.task_name,
      -- 1) pack into [(name,value),…]
      -- 2) arraySort by name
      -- 3) arrayMap to pull out names
      arrayMap(
        t -> t.1,
        arraySort(
          (t) -> t.1,
          groupArrayIf((f.metric_name, f.value), f.metric_name != '')
        )
      ) AS feedback_metric_names,

      -- same 3-step, but pull out values
      arrayMap(
        t -> t.2,
        arraySort(
          (t) -> t.1,
          groupArrayIf((f.metric_name, f.value), f.metric_name != '')
        )
      ) AS feedback_values

    FROM episodes AS e
    LEFT JOIN feedback_union AS f
      ON f.target_id = uint_to_uuid(e.episode_id_uint)
    GROUP BY
      e.episode_id_uint,
      e.run_id_uint,
      e.tags,
      e.task_name
  `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { page_size, offset, run_id },
  });
  const rows = await result.json<DynamicEvaluationRunEpisodeWithFeedback[]>();
  return rows.map((row) =>
    dynamicEvaluationRunEpisodeWithFeedbackSchema.parse(row),
  );
}

export async function getDynamicEvaluationRunStatisticsByMetricName(
  run_id: string,
  metric_name?: string,
): Promise<DynamicEvaluationRunStatisticsByMetricName[]> {
  const query = `
    WITH
       episodes AS (
        SELECT
          episode_id_uint,
          run_id_uint,
          tags,
          datapoint_name
        FROM DynamicEvaluationRunEpisodeByRunId
        WHERE toUInt128(toUUID({run_id:String})) = run_id_uint
        ORDER BY episode_id_uint DESC
      )
    SELECT
      metric_name,
      toUInt32(count()) as count,
      avg(value) as avg_metric,
      stddevSamp(value) as stdev,
      1.96 * (stddevSamp(value) / sqrt(count())) AS ci_error
    FROM FloatMetricFeedbackByTargetId
    WHERE target_id IN (
      SELECT uint_to_uuid(episode_id_uint) FROM episodes
    )
    ${metric_name ? `AND metric_name = {metric_name:String}` : ""}
    GROUP BY metric_name
    UNION ALL
    SELECT
      metric_name,
      toUInt32(count()) as count,
      avg(value) as avg_metric,
      stddevSamp(value) as stdev,
      1.96 * (stddevSamp(value) / sqrt(count())) AS ci_error
    FROM BooleanMetricFeedbackByTargetId
    WHERE target_id IN (
      SELECT uint_to_uuid(episode_id_uint) FROM episodes
    )
    ${metric_name ? `AND metric_name = {metric_name:String}` : ""}
    GROUP BY metric_name
    ORDER BY metric_name ASC
  `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { run_id, metric_name },
  });
  const rows =
    await result.json<DynamicEvaluationRunStatisticsByMetricName[]>();
  return rows.map((row) =>
    dynamicEvaluationRunStatisticsByMetricNameSchema.parse(row),
  );
}

export async function countDynamicEvaluationRunEpisodes(
  run_id: string,
): Promise<number> {
  const query = `
    SELECT toUInt32(count()) as count FROM DynamicEvaluationRunEpisodeByRunId
    WHERE toUInt128(toUUID({run_id:String})) = run_id_uint
  `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { run_id },
  });
  const rows = await result.json<{ count: number }>();
  return rows[0].count;
}
