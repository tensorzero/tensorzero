import { getClickhouseClient } from "./client.server";
import { CountSchema } from "./common";
import {
  workflowEvaluationProjectSchema,
  workflowEvaluationRunEpisodeWithFeedbackSchema,
  workflowEvaluationRunSchema,
  workflowEvaluationRunStatisticsByMetricNameSchema,
  groupedWorkflowEvaluationRunEpisodeWithFeedbackSchema,
  type WorkflowEvaluationProject,
  type WorkflowEvaluationRun,
  type WorkflowEvaluationRunEpisodeWithFeedback,
  type WorkflowEvaluationRunStatisticsByMetricName,
  type GroupedWorkflowEvaluationRunEpisodeWithFeedback,
  type WorkflowEvaluationRunWithEpisodeCount,
  workflowEvaluationRunWithEpisodeCountSchema,
} from "./workflow_evaluations";

export async function getWorkflowEvaluationRuns(
  page_size: number,
  offset: number,
  run_id?: string,
  project_name?: string,
): Promise<WorkflowEvaluationRunWithEpisodeCount[]> {
  const query = `
    WITH FilteredWorkflowEvaluationRuns AS (
      SELECT
          run_display_name as name,
          uint_to_uuid(run_id_uint) as id,
          run_id_uint,
          variant_pins,
          tags,
          project_name,
          formatDateTime(UUIDv7ToDateTime(uint_to_uuid(run_id_uint)), '%Y-%m-%dT%H:%i:%SZ') as timestamp
      FROM DynamicEvaluationRun
      ${run_id ? `WHERE toUInt128(toUUID({run_id:String})) = run_id_uint` : ""}
      ${project_name ? `WHERE project_name = {project_name:String}` : ""}
      ORDER BY run_id_uint DESC
      LIMIT {page_size:UInt64}
      OFFSET {offset:UInt64}
    ),
    WorkflowEvaluationRunsEpisodeCounts AS (
      SELECT
        run_id_uint,
        toUInt32(count()) as num_episodes
      FROM DynamicRunEpisodeByRunId
      WHERE run_id_uint IN (SELECT run_id_uint FROM FilteredWorkflowEvaluationRuns)
      GROUP BY run_id_uint
    )
    SELECT
      name,
      id,
      variant_pins,
      tags,
      project_name,
      COALESCE(num_episodes, 0) AS num_episodes,
      timestamp
    FROM FilteredWorkflowEvaluationRuns
    LEFT JOIN WorkflowEvaluationRunsEpisodeCounts USING run_id_uint
    ORDER BY run_id_uint DESC
  `;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      page_size,
      offset,
      run_id,
      project_name,
    },
  });
  const rows = await result.json<WorkflowEvaluationRunWithEpisodeCount[]>();
  return rows.map((row) =>
    workflowEvaluationRunWithEpisodeCountSchema.parse(row),
  );
}

export async function getWorkflowEvaluationRunsByIds(
  run_ids: string[], // one or more UUIDv7 strings
  project_name?: string, // optional extra filter
): Promise<WorkflowEvaluationRun[]> {
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

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { run_ids, project_name },
  });

  const rows = await result.json<WorkflowEvaluationRun[]>();
  return rows.map((row) => workflowEvaluationRunSchema.parse(row));
}

export async function countWorkflowEvaluationRuns(): Promise<number> {
  const query = `
    SELECT toUInt32(count()) as count FROM DynamicEvaluationRun
  `;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
  });
  const rows = await result.json<{ count: number }[]>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}

/**
 * Returns information about the episodes that were used in a workflow evaluation run,
 * along with the feedback that was collected for each episode.
 *
 * The feedback is given as arrays feedback_metric_names and feedback_values.
 * The arrays are sorted by the metric name.
 */
export async function getWorkflowEvaluationRunEpisodesByRunIdWithFeedback(
  page_size: number,
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
        FROM DynamicRunEpisodeByRunId
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
    ORDER BY
        e.episode_id_uint DESC
  `;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { page_size, offset, run_id },
  });
  const rows = await result.json<WorkflowEvaluationRunEpisodeWithFeedback[]>();
  return rows.map((row) =>
    workflowEvaluationRunEpisodeWithFeedbackSchema.parse(row),
  );
}

export async function getWorkflowEvaluationRunStatisticsByMetricName(
  run_id: string,
  metric_name?: string,
): Promise<WorkflowEvaluationRunStatisticsByMetricName[]> {
  const query = `
    WITH
       episodes AS (
        SELECT
          episode_id_uint,
          run_id_uint,
          tags,
          datapoint_name -- for legacy reasons, \`task_name\` is stored as \`datapoint_name\` in the database
        FROM DynamicRunEpisodeByRunId
        WHERE toUInt128(toUUID({run_id:String})) = run_id_uint
        ORDER BY episode_id_uint DESC
      ),
    results AS (
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
    )
    SELECT * FROM results
    ORDER BY metric_name ASC
  `;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { run_id, metric_name },
  });
  const rows =
    await result.json<WorkflowEvaluationRunStatisticsByMetricName[]>();
  return rows.map((row) =>
    workflowEvaluationRunStatisticsByMetricNameSchema.parse(row),
  );
}

export async function countWorkflowEvaluationRunEpisodes(
  run_id: string,
): Promise<number> {
  const query = `
    SELECT toUInt32(count()) as count FROM DynamicRunEpisodeByRunId
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

export async function getWorkflowEvaluationProjects(
  page_size: number,
  offset: number,
): Promise<WorkflowEvaluationProject[]> {
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
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { page_size, offset },
  });
  const rows = await result.json<WorkflowEvaluationProject[]>();
  return rows.map((row) => workflowEvaluationProjectSchema.parse(row));
}

export async function countWorkflowEvaluationProjects(): Promise<number> {
  const query = `
  SELECT toUInt32(countDistinct(project_name)) AS count
  FROM DynamicEvaluationRunByProjectName
  WHERE project_name IS NOT NULL
`;
  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
  });
  const rows = await result.json<{ count: number }[]>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}

export async function searchWorkflowEvaluationRuns(
  page_size: number,
  offset: number,
  project_name?: string,
  search_query?: string,
): Promise<WorkflowEvaluationRun[]> {
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

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { project_name, search_query, page_size, offset },
  });
  const rows = await result.json<WorkflowEvaluationRun[]>();
  return rows.map((row) => workflowEvaluationRunSchema.parse(row));
}

/**
 * Returns a list of episodes that were part of some set of workflow evaluation runs,
 * grouped into sublists that all have the same task_name.
 * If the task_name is NULL, the episode is grouped into a sublist by itself.
 *
 * The returned list is sorted by the timestamp of the episode.
 */
export async function getWorkflowEvaluationRunEpisodesByTaskName(
  runIds: string[],
  page_size: number,
  offset: number,
): Promise<GroupedWorkflowEvaluationRunEpisodeWithFeedback[][]> {
  const query = `
    WITH
      -- 1) pull all episodes for these runIds
      episodes_raw AS (
        SELECT
          episode_id_uint,
          run_id_uint,
          tags,
          updated_at,
          datapoint_name AS task_name -- for legacy reasons, \`task_name\` is stored as \`datapoint_name\` in the database
        FROM DynamicEvaluationRunEpisodeByRunId
        WHERE run_id_uint IN (
          SELECT arrayJoin(
            arrayMap(x -> toUInt128(toUUID(x)), {runIds:Array(String)})
          )
        )
      ),

      -- 2) pick out the distinct group_keys, page them
      group_keys AS (
        SELECT
          ifNull(task_name, concat('NULL_EPISODE_', toString(episode_id_uint))) AS group_key,
          max(updated_at) as last_updated
        FROM episodes_raw
        GROUP BY group_key
        ORDER BY last_updated DESC
        LIMIT {page_size:UInt64}
        OFFSET {offset:UInt64}
      ),

      -- 3) only keep episodes whose group_key made the cut
      episodes AS (
        SELECT
          *,
          ifNull(task_name, concat('NULL_EPISODE_', toString(episode_id_uint))) AS group_key
        FROM episodes_raw
        WHERE ifNull(task_name, concat('NULL_EPISODE_', toString(episode_id_uint)))
          IN (SELECT group_key FROM group_keys)
      ),

      -- 4) gather feedback just for those episodes
      feedback_union AS (
        SELECT target_id, metric_name,
               argMax(toString(value), toUInt128(id)) AS value
        FROM FloatMetricFeedbackByTargetId
        WHERE target_id IN (
          SELECT uint_to_uuid(episode_id_uint) FROM episodes
        )
        GROUP BY target_id, metric_name

        UNION ALL

        SELECT target_id, metric_name,
               argMax(toString(value), toUInt128(id)) AS value
        FROM BooleanMetricFeedbackByTargetId
        WHERE target_id IN (
          SELECT uint_to_uuid(episode_id_uint) FROM episodes
        )
        GROUP BY target_id, metric_name

        UNION ALL

        SELECT target_id,
               'comment' AS metric_name,
               value
        FROM CommentFeedbackByTargetId
        WHERE target_id IN (
          SELECT uint_to_uuid(episode_id_uint) FROM episodes
        )
      )

    SELECT
      e.group_key as group_key,
      uint_to_uuid(e.episode_id_uint) AS episode_id,
      formatDateTime(min(e.updated_at), '%Y-%m-%dT%H:%i:%SZ') AS timestamp,
      uint_to_uuid(e.run_id_uint) AS run_id,
      e.tags,
      e.task_name,
      arrayMap(t -> t.1,
        arraySort((t)->t.1,
          groupArrayIf((f.metric_name,f.value), f.metric_name!='')
        )
      ) AS feedback_metric_names,
      arrayMap(t -> t.2,
        arraySort((t)->t.1,
          groupArrayIf((f.metric_name,f.value), f.metric_name!='')
        )
      ) AS feedback_values
    FROM episodes AS e
    -- rejoin the group_keys CTE to get the last_updated timestamp
    JOIN group_keys AS g USING group_key
    LEFT JOIN feedback_union AS f
      ON f.target_id = uint_to_uuid(e.episode_id_uint)
    GROUP BY
      e.group_key,
      e.episode_id_uint,
      e.run_id_uint,
      e.tags,
      e.task_name
    ORDER BY
      max(g.last_updated) DESC,
      e.group_key,
      e.episode_id_uint
  `;

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { runIds, page_size, offset },
  });
  const raw =
    await result.json<GroupedWorkflowEvaluationRunEpisodeWithFeedback>();

  // bucket by group_key, parse each row with your Zod schema
  const buckets: Record<
    string,
    GroupedWorkflowEvaluationRunEpisodeWithFeedback[]
  > = {};
  for (const row of raw) {
    const eps =
      groupedWorkflowEvaluationRunEpisodeWithFeedbackSchema.parse(row);
    buckets[eps.group_key] ??= [];
    buckets[eps.group_key].push(eps);
  }

  // return an array of episode‑arrays, one per group
  return Object.values(buckets);
}

/**
 * Counts the number of groups that would be returned by getWorkflowEvaluationRunEpisodesByTaskName with no pagination.
 */
export async function countWorkflowEvaluationRunEpisodesByTaskName(
  runIds: string[],
): Promise<number> {
  const query = `
    WITH
      episodes_raw AS (
        SELECT
          episode_id_uint,
          run_id_uint,
          tags,
          updated_at,
          datapoint_name AS task_name -- for legacy reasons, \`task_name\` is stored as \`datapoint_name\` in the database
        FROM DynamicEvaluationRunEpisodeByRunId
        WHERE run_id_uint IN (
          SELECT arrayJoin(
            arrayMap(x -> toUInt128(toUUID(x)), {runIds:Array(String)})
          )
        )
      )
    SELECT toUInt32(countDistinct(ifNull(task_name, concat('NULL_EPISODE_', toString(episode_id_uint))))) as count
    FROM episodes_raw
  `;

  const result = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { runIds },
  });
  const rows = await result.json<{ count: number }[]>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}
