import { clickhouseClient } from "./client.server";
import {
  dynamicEvaluationProjectSchema,
  dynamicEvaluationRunEpisodeWithFeedbackSchema,
  dynamicEvaluationRunSchema,
  dynamicEvaluationRunStatisticsByMetricNameSchema,
  type DynamicEvaluationProject,
  type DynamicEvaluationRun,
  type DynamicEvaluationRunEpisodeWithFeedback,
  type DynamicEvaluationRunStatisticsByMetricName,
} from "./dynamic_evaluations";

export async function getDynamicEvaluationRuns(
  page_size: number,
  offset: number,
  run_id?: string,
  project_name?: string,
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
    ${project_name ? `WHERE project_name = {project_name:String}` : ""}
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
      project_name,
    },
  });
  const rows = await result.json<DynamicEvaluationRun[]>();
  return rows.map((row) => dynamicEvaluationRunSchema.parse(row));
}

export async function getDynamicEvaluationRunsByIds(
  run_ids: string[], // one or more UUIDv7 strings
  project_name?: string, // optional extra filter
): Promise<DynamicEvaluationRun[]> {
  if (run_ids.length === 0) return []; // nothing to fetch

  const query = `
    SELECT
      run_display_name AS name,
      uint_to_uuid(run_id_uint) AS id,
      variant_pins,
      tags,
      project_name,
      formatDateTime(
        UUIDv7ToDateTime(uint_to_uuid(run_id_uint)),
        '%Y-%m-%dT%H:%i:%SZ'
      ) AS timestamp
    FROM DynamicEvaluationRun
    WHERE run_id_uint IN (
      /* turn the parameter array of UUID strings into a real table
         expression of UInt128 values so the IN predicate is valid */
      SELECT arrayJoin(
        arrayMap(x -> toUInt128(toUUID(x)), {run_ids:Array(String)})
      )
    )
    ${project_name ? "AND project_name = {project_name:String}" : ""}
    ORDER BY run_id_uint DESC
  `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { run_ids, project_name },
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

export async function getDynamicEvaluationProjects(
  page_size: number,
  offset: number,
): Promise<DynamicEvaluationProject[]> {
  const query = `
    SELECT
      project_name as name,
      toUInt32(count()) as count,
      formatDateTime(max(updated_at), '%Y-%m-%dT%H:%i:%SZ')  as last_updated
    FROM DynamicEvaluationRunByProjectName
    GROUP BY project_name
    ORDER BY last_updated DESC
    LIMIT {page_size:UInt64}
    OFFSET {offset:UInt64}
  `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { page_size, offset },
  });
  const rows = await result.json<DynamicEvaluationProject[]>();
  return rows.map((row) => dynamicEvaluationProjectSchema.parse(row));
}

export async function countDynamicEvaluationProjects(): Promise<number> {
  const query = `
  SELECT toUInt32(countDistinct(project_name)) AS count
  FROM DynamicEvaluationRunByProjectName
  WHERE project_name IS NOT NULL
`;
  const result = await clickhouseClient.query({ query, format: "JSONEachRow" });
  const rows = await result.json<{ count: number }>();
  return rows[0].count;
}

export async function searchDynamicEvaluationRuns(
  page_size: number,
  offset: number,
  project_name?: string,
  search_query?: string,
): Promise<DynamicEvaluationRun[]> {
  // 1) Build an array of individual predicates
  const predicates: string[] = [];

  if (project_name) {
    predicates.push(`project_name = {project_name:String}`);
  }

  if (search_query) {
    predicates.push(`(
      positionCaseInsensitive(run_display_name, {search_query:String}) > 0
      OR positionCaseInsensitive(toString(uint_to_uuid(run_id_uint)), {search_query:String}) > 0
    )`);
  }

  // 2) If we have any predicates, join them with " AND " and prefix with WHERE
  const whereClause = predicates.length
    ? `WHERE ${predicates.join(" AND ")}`
    : "";

  // 3) Plug the one WHERE (or nothing) into your template
  const query = `
    SELECT
      run_display_name as name,
      uint_to_uuid(run_id_uint) as id,
      variant_pins,
      tags,
      project_name,
      formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') as timestamp
    FROM DynamicEvaluationRun
    ${whereClause}
    ORDER BY updated_at DESC
    LIMIT {page_size:UInt64}
    OFFSET {offset:UInt64}
  `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { project_name, search_query, page_size, offset },
  });
  const rows = await result.json<DynamicEvaluationRun[]>();
  return rows.map((row) => dynamicEvaluationRunSchema.parse(row));
}

export async function getDynamicEvaluationRunByDatapointName(
  runIds: string[],
  page_size: number,
  offset: number,
): Promise<DynamicEvaluationRun[]> {
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
        WHERE run_id_uint IN (
          SELECT arrayJoin(arrayMap(x -> toUInt128(toUUID(x)), {runIds:Array(String)}))
        )
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
      ),
    grouped AS (
      SELECT
          -- If task_name is NULL, group by episode_id_uint to treat each as singleton
          -- Else use the task_name
          ifNull(task_name, concat('NULL_EPISODE_', toString(episode_id_uint))) as group_key,
          any(task_name) as task_name,
          max(updated_at) as max_updated_at,
          any(episode_id_uint) as episode_id_uint,
          any(run_id_uint) as run_id_uint,
          any(tags) as tags
        FROM episodes
        GROUP BY group_key
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
    query_params: { page_size, offset, runIds },
  });
  const rows = await result.json<DynamicEvaluationRunEpisodeWithFeedback[]>();
  return rows.map((row) =>
    dynamicEvaluationRunEpisodeWithFeedbackSchema.parse(row),
  );
}
