import { getClickhouseClient } from "./client.server";
import {
  workflowEvaluationRunEpisodeWithFeedbackSchema,
  type WorkflowEvaluationRunEpisodeWithFeedback,
} from "./workflow_evaluations";

/**
 * Returns information about the episodes that were used in a workflow evaluation run,
 * along with the feedback that was collected for each episode.
 *
 * The feedback is given as arrays feedback_metric_names and feedback_values.
 * The arrays are sorted by the metric name.
 */
export async function getWorkflowEvaluationRunEpisodesByRunIdWithFeedback(
  limit: number,
  offset: number,
  run_id: string,
): Promise<WorkflowEvaluationRunEpisodeWithFeedback[]> {
  const query = `
    WITH
      episodes AS (
        SELECT
          episode_id_uint,
          run_id_uint,
          tags,
          updated_at,
          datapoint_name AS task_name, -- for legacy reasons, \`task_name\` is stored as \`datapoint_name\` in the database
          ifNull(datapoint_name, concat('NULL_EPISODE_', toString(episode_id_uint))) as group_key
        FROM DynamicEvaluationRunEpisodeByRunId
        WHERE toUInt128(toUUID({run_id:String})) = run_id_uint
        ORDER BY episode_id_uint DESC
        LIMIT {limit:UInt64}
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
    ORDER BY
        e.episode_id_uint DESC
  `;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { limit, offset, run_id },
  });
  const rows = await result.json<WorkflowEvaluationRunEpisodeWithFeedback[]>();
  return rows.map((row) =>
    workflowEvaluationRunEpisodeWithFeedbackSchema.parse(row),
  );
}

export async function countWorkflowEvaluationRunEpisodes(
  run_id: string,
): Promise<number> {
  const query = `
    SELECT toUInt32(count()) as count FROM DynamicEvaluationRunEpisodeByRunId
    WHERE toUInt128(toUUID({run_id:String})) = run_id_uint
  `;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { run_id },
  });
  const rows = await result.json<{ count: number }>();
  return rows[0].count;
}
