//! ClickHouse queries for workflow evaluation statistics.

use std::collections::HashMap;

use async_trait::async_trait;
use uuid::Uuid;

use super::select_queries::{parse_count, parse_json_rows};
use super::{ClickHouseConnectionInfo, escape_string_for_clickhouse_literal};
use crate::config::snapshot::SnapshotHash;
use crate::db::workflow_evaluation_queries::{
    GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow, WorkflowEvaluationProjectRow,
    WorkflowEvaluationQueries, WorkflowEvaluationRunEpisodeWithFeedbackRow,
    WorkflowEvaluationRunInfo, WorkflowEvaluationRunRow, WorkflowEvaluationRunStatisticsRaw,
    WorkflowEvaluationRunStatisticsRow, WorkflowEvaluationRunWithEpisodeCountRow,
};
use crate::error::{Error, ErrorDetails};
use crate::statistics_util::{wald_confint, wilson_confint};

#[async_trait]
impl WorkflowEvaluationQueries for ClickHouseConnectionInfo {
    async fn list_workflow_evaluation_projects(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<WorkflowEvaluationProjectRow>, Error> {
        let query = r"
            SELECT
                project_name as name,
                toUInt32(count()) as count,
                formatDateTime(max(updated_at), '%Y-%m-%dT%H:%i:%SZ') as last_updated
            FROM DynamicEvaluationRunByProjectName
            GROUP BY project_name
            ORDER BY last_updated DESC
            LIMIT {limit:UInt32}
            OFFSET {offset:UInt32}
            FORMAT JSONEachRow
        "
        .to_string();

        let limit_str = limit.to_string();
        let offset_str = offset.to_string();
        let mut params = HashMap::new();
        params.insert("limit", limit_str.as_str());
        params.insert("offset", offset_str.as_str());

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn count_workflow_evaluation_projects(&self) -> Result<u32, Error> {
        let query = r"
            SELECT
                toUInt32(countDistinct(project_name)) as count
            FROM DynamicEvaluationRunByProjectName
            WHERE project_name IS NOT NULL
            FORMAT JSONEachRow
        "
        .to_string();

        let response = self.run_query_synchronous_no_params(query).await?;
        let count = parse_count(response.response.as_str())?;

        u32::try_from(count).map_err(|error| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: format!("Failed to convert workflow evaluation project count: {error}"),
            })
        })
    }

    async fn search_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        project_name: Option<&str>,
        search_query: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error> {
        // Build WHERE clause predicates
        let mut predicates: Vec<String> = Vec::new();

        if project_name.is_some() {
            predicates.push("project_name = {project_name:String}".to_string());
        }

        if search_query.is_some() {
            predicates.push(
                "(positionCaseInsensitive(run_display_name, {search_query:String}) > 0 \
                 OR positionCaseInsensitive(toString(uint_to_uuid(run_id_uint)), {search_query:String}) > 0)"
                    .to_string(),
            );
        }

        let where_clause = if predicates.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", predicates.join(" AND "))
        };

        let query = format!(
            r"
            SELECT
                run_display_name as name,
                uint_to_uuid(run_id_uint) as id,
                variant_pins,
                tags,
                project_name,
                formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') as timestamp
            FROM DynamicEvaluationRun
            {where_clause}
            ORDER BY updated_at DESC
            LIMIT {{limit:UInt32}}
            OFFSET {{offset:UInt32}}
            FORMAT JSONEachRow
            "
        );

        let limit_str = limit.to_string();
        let offset_str = offset.to_string();
        let project_name_str = project_name.unwrap_or_default().to_string();
        let search_query_str = search_query.unwrap_or_default().to_string();

        let mut params = HashMap::new();
        params.insert("limit", limit_str.as_str());
        params.insert("offset", offset_str.as_str());
        if project_name.is_some() {
            params.insert("project_name", project_name_str.as_str());
        }
        if search_query.is_some() {
            params.insert("search_query", search_query_str.as_str());
        }

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn list_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        run_id: Option<Uuid>,
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunWithEpisodeCountRow>, Error> {
        // Build WHERE clause - only one of run_id or project_name should be provided
        let where_clause = if run_id.is_some() {
            "WHERE toUInt128(toUUID({run_id:String})) = run_id_uint"
        } else if project_name.is_some() {
            "WHERE project_name = {project_name:String}"
        } else {
            ""
        };

        let query = format!(
            r"
            WITH FilteredDynamicEvaluationRuns AS (
                SELECT
                    run_display_name as name,
                    uint_to_uuid(run_id_uint) as id,
                    run_id_uint,
                    variant_pins,
                    tags,
                    project_name,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(run_id_uint)), '%Y-%m-%dT%H:%i:%SZ') as timestamp
                FROM DynamicEvaluationRun
                {where_clause}
                ORDER BY run_id_uint DESC
                LIMIT {{limit:UInt32}}
                OFFSET {{offset:UInt32}}
            ),
            DynamicEvaluationRunsEpisodeCounts AS (
                SELECT
                    run_id_uint,
                    toUInt32(count()) as num_episodes
                FROM DynamicEvaluationRunEpisodeByRunId
                WHERE run_id_uint IN (SELECT run_id_uint FROM FilteredDynamicEvaluationRuns)
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
            FROM FilteredDynamicEvaluationRuns
            LEFT JOIN DynamicEvaluationRunsEpisodeCounts USING run_id_uint
            ORDER BY run_id_uint DESC
            FORMAT JSONEachRow
            "
        );

        let limit_str = limit.to_string();
        let offset_str = offset.to_string();
        let run_id_str = run_id.map(|id| id.to_string()).unwrap_or_default();
        let project_name_str = project_name.unwrap_or_default().to_string();

        let mut params = HashMap::new();
        params.insert("limit", limit_str.as_str());
        params.insert("offset", offset_str.as_str());
        if run_id.is_some() {
            params.insert("run_id", run_id_str.as_str());
        }
        if project_name.is_some() {
            params.insert("project_name", project_name_str.as_str());
        }

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn count_workflow_evaluation_runs(&self) -> Result<u32, Error> {
        let query = r"
            SELECT toUInt32(count()) as count FROM DynamicEvaluationRun
            FORMAT JSONEachRow
        "
        .to_string();

        let response = self.run_query_synchronous_no_params(query).await?;
        let count = parse_count(response.response.as_str())?;

        u32::try_from(count).map_err(|error| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: format!("Failed to convert workflow evaluation run count: {error}"),
            })
        })
    }

    async fn get_workflow_evaluation_runs(
        &self,
        run_ids: &[Uuid],
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error> {
        if run_ids.is_empty() {
            return Ok(vec![]);
        }

        let mut params = HashMap::new();

        let project_name_filter = if let Some(project_name_str) = project_name {
            params.insert("project_name", project_name_str);
            "AND project_name = {project_name:String}"
        } else {
            ""
        };

        let query = format!(
            r"
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
                SELECT arrayJoin(
                    arrayMap(x -> toUInt128(toUUID(x)), {{run_ids:Array(String)}})
                )
            )
            {project_name_filter}
            ORDER BY run_id_uint DESC
            FORMAT JSONEachRow
            "
        );

        // Format run_ids as ClickHouse array with single quotes
        let run_ids_str: Vec<String> = run_ids.iter().map(|id| format!("'{id}'")).collect();
        let run_ids_array = format!("[{}]", run_ids_str.join(","));
        params.insert("run_ids", run_ids_array.as_str());

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn get_workflow_evaluation_run_statistics(
        &self,
        run_id: Uuid,
        metric_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunStatisticsRow>, Error> {
        let run_id_str = run_id.to_string();

        let mut params = HashMap::new();
        params.insert("run_id", run_id_str.as_str());

        // Build the optional metric name filter
        let metric_name_filter = if let Some(metric_name) = metric_name {
            params.insert("metric_name", metric_name);
            "AND metric_name = {metric_name:String}"
        } else {
            ""
        };

        let query = format!(
            r"
            WITH
              episodes AS (
                SELECT episode_id_uint
                FROM DynamicEvaluationRunEpisodeByRunId
                WHERE toUInt128(toUUID({{run_id:String}})) = run_id_uint
              ),
              float_stats AS (
                SELECT
                  metric_name,
                  toUInt32(count()) as count,
                  avg(value) as avg_metric,
                  stddevSamp(value) as stdev,
                  false as is_boolean
                FROM FloatMetricFeedbackByTargetId
                WHERE target_id IN (
                  SELECT uint_to_uuid(episode_id_uint) FROM episodes
                )
                {metric_name_filter}
                GROUP BY metric_name
              ),
              boolean_stats AS (
                SELECT
                  metric_name,
                  toUInt32(count()) as count,
                  avg(value) as avg_metric,
                  stddevSamp(value) as stdev,
                  true as is_boolean
                FROM BooleanMetricFeedbackByTargetId
                WHERE target_id IN (
                  SELECT uint_to_uuid(episode_id_uint) FROM episodes
                )
                {metric_name_filter}
                GROUP BY metric_name
              )
            SELECT * FROM float_stats
            UNION ALL
            SELECT * FROM boolean_stats
            ORDER BY metric_name ASC
            FORMAT JSONEachRow
            "
        );

        let response = self.run_query_synchronous(query, &params).await?;
        let raw_stats: Vec<WorkflowEvaluationRunStatisticsRaw> =
            parse_json_rows(response.response.as_str())?;

        // Compute confidence intervals in Rust
        let stats = raw_stats
            .into_iter()
            .map(|raw| {
                let ci = if raw.is_boolean {
                    wilson_confint(raw.avg_metric, raw.count)
                } else if let Some(stdev) = raw.stdev {
                    wald_confint(raw.avg_metric, stdev, raw.count)
                } else {
                    None
                };

                let (ci_lower, ci_upper) = match ci {
                    Some((lower, upper)) => (Some(lower), Some(upper)),
                    None => (None, None),
                };

                WorkflowEvaluationRunStatisticsRow {
                    metric_name: raw.metric_name,
                    count: raw.count,
                    avg_metric: raw.avg_metric,
                    stdev: raw.stdev,
                    ci_lower,
                    ci_upper,
                }
            })
            .collect();

        Ok(stats)
    }

    async fn list_workflow_evaluation_run_episodes_by_task_name(
        &self,
        run_ids: &[Uuid],
        limit: u32,
        offset: u32,
    ) -> Result<Vec<GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow>, Error> {
        if run_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Convert run_ids to a comma-separated string of quoted UUIDs for the query
        let run_ids_array: Vec<String> = run_ids.iter().map(|id| format!("'{id}'")).collect();
        let run_ids_str = run_ids_array.join(",");

        let query = format!(
            r"
            WITH
              -- 1) pull all episodes for these runIds
              episodes_raw AS (
                SELECT
                  episode_id_uint,
                  run_id_uint,
                  tags,
                  updated_at,
                  -- for legacy reasons, `task_name` is stored as `datapoint_name`
                  datapoint_name AS task_name
                FROM DynamicEvaluationRunEpisodeByRunId
                WHERE run_id_uint IN (
                  SELECT arrayJoin(
                    arrayMap(x -> toUInt128(toUUID(x)), [{run_ids_str}])
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
                LIMIT {{limit:UInt32}}
                OFFSET {{offset:UInt32}}
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
            FORMAT JSONEachRow"
        );

        let limit_str = limit.to_string();
        let offset_str = offset.to_string();

        let mut params = HashMap::new();
        params.insert("limit", limit_str.as_str());
        params.insert("offset", offset_str.as_str());

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn count_workflow_evaluation_run_episodes_by_task_name(
        &self,
        run_ids: &[Uuid],
    ) -> Result<u32, Error> {
        if run_ids.is_empty() {
            return Ok(0);
        }

        // Convert run_ids to a comma-separated string of quoted UUIDs for the query
        let run_ids_array: Vec<String> = run_ids.iter().map(|id| format!("'{id}'")).collect();
        let run_ids_str = run_ids_array.join(",");

        let query = format!(
            r"
            WITH
              episodes_raw AS (
                SELECT
                  episode_id_uint,
                  run_id_uint,
                  tags,
                  updated_at,
                  datapoint_name AS task_name
                FROM DynamicEvaluationRunEpisodeByRunId
                WHERE run_id_uint IN (
                  SELECT arrayJoin(
                    arrayMap(x -> toUInt128(toUUID(x)), [{run_ids_str}])
                  )
                )
              )
            SELECT toUInt32(countDistinct(ifNull(task_name, concat('NULL_EPISODE_', toString(episode_id_uint))))) as count
            FROM episodes_raw
            FORMAT JSONEachRow"
        );

        let response = self.run_query_synchronous_no_params(query).await?;
        let count = parse_count(response.response.as_str())?;

        u32::try_from(count).map_err(|error| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: format!(
                    "Failed to convert workflow evaluation episode group count: {error}"
                ),
            })
        })
    }

    async fn get_workflow_evaluation_run_episodes_with_feedback(
        &self,
        run_id: Uuid,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<WorkflowEvaluationRunEpisodeWithFeedbackRow>, Error> {
        let query = r"
            WITH
              episodes AS (
                SELECT
                  episode_id_uint,
                  run_id_uint,
                  tags,
                  updated_at,
                  datapoint_name AS task_name
                FROM DynamicEvaluationRunEpisodeByRunId
                WHERE toUInt128(toUUID({run_id:String})) = run_id_uint
                ORDER BY episode_id_uint DESC
                LIMIT {limit:UInt32}
                OFFSET {offset:UInt32}
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
                min(e.updated_at),
                '%Y-%m-%dT%H:%i:%SZ'
              ) AS timestamp,
              uint_to_uuid(e.run_id_uint) AS run_id,
              e.tags,
              e.task_name,
              arrayMap(
                t -> t.1,
                arraySort(
                  (t) -> t.1,
                  groupArrayIf((f.metric_name, f.value), f.metric_name != '')
                )
              ) AS feedback_metric_names,
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
            FORMAT JSONEachRow
        "
        .to_string();

        let run_id_str = run_id.to_string();
        let limit_str = limit.to_string();
        let offset_str = offset.to_string();

        let mut params = HashMap::new();
        params.insert("run_id", run_id_str.as_str());
        params.insert("limit", limit_str.as_str());
        params.insert("offset", offset_str.as_str());

        let response = self.run_query_synchronous(query, &params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn count_workflow_evaluation_run_episodes(&self, run_id: Uuid) -> Result<u32, Error> {
        let query = r"
            SELECT toUInt32(count()) as count
            FROM DynamicEvaluationRunEpisodeByRunId
            WHERE toUInt128(toUUID({run_id:String})) = run_id_uint
            FORMAT JSONEachRow
        "
        .to_string();

        let run_id_str = run_id.to_string();
        let mut params = HashMap::new();
        params.insert("run_id", run_id_str.as_str());

        let response = self.run_query_synchronous(query, &params).await?;
        let count = parse_count(response.response.as_str())?;

        u32::try_from(count).map_err(|error| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: format!(
                    "Failed to convert workflow evaluation run episode count: {error}"
                ),
            })
        })
    }

    async fn insert_workflow_evaluation_run(
        &self,
        run_id: Uuid,
        variant_pins: &HashMap<String, String>,
        tags: &HashMap<String, String>,
        project_name: Option<&str>,
        run_display_name: Option<&str>,
        snapshot_hash: &SnapshotHash,
    ) -> Result<(), Error> {
        let query = r"
        INSERT INTO DynamicEvaluationRun (
            run_id_uint,
            variant_pins,
            tags,
            project_name,
            run_display_name,
            snapshot_hash
        )
        VALUES (
            toUInt128({run_id:UUID}),
            {variant_pins:Map(String, String)},
            {tags:Map(String, String)},
            {project_name:Nullable(String)},
            {run_display_name:Nullable(String)},
            toUInt256OrNull({snapshot_hash:Nullable(String)})
        )
        ";

        let run_id_str = run_id.to_string();
        let variant_pins_str = to_map_literal(variant_pins);
        let tags_str = to_map_literal(tags);

        let mut params = HashMap::new();
        params.insert("run_id", run_id_str.as_str());
        params.insert("variant_pins", variant_pins_str.as_str());
        params.insert("tags", tags_str.as_str());
        // Use \\N to indicate NULL
        params.insert("project_name", project_name.unwrap_or("\\N"));
        params.insert("run_display_name", run_display_name.unwrap_or("\\N"));
        params.insert("snapshot_hash", &**snapshot_hash);

        self.run_query_synchronous(query.to_string(), &params)
            .await?;
        Ok(())
    }

    async fn insert_workflow_evaluation_run_episode(
        &self,
        run_id: Uuid,
        episode_id: Uuid,
        task_name: Option<&str>,
        tags: &HashMap<String, String>,
        snapshot_hash: &SnapshotHash,
    ) -> Result<(), Error> {
        let query = r"
        INSERT INTO DynamicEvaluationRunEpisode
        (
            run_id,
            episode_id_uint,
            variant_pins,
            datapoint_name, -- for legacy reasons, `task_name` is stored as `datapoint_name` in the database
            tags,
            snapshot_hash
        )
        SELECT
            {run_id:UUID} AS run_id,
            toUInt128({episode_id:UUID}) AS episode_id_uint,
            variant_pins,
            {datapoint_name:Nullable(String)} AS datapoint_name, -- for legacy reasons, `task_name` is stored as `datapoint_name` in the database
            mapUpdate(tags, {tags:Map(String, String)}) AS tags, -- merge the tags in the params on top of tags in the workflow evaluation run
            toUInt256OrNull({snapshot_hash:Nullable(String)}) AS snapshot_hash
        FROM DynamicEvaluationRun
        WHERE run_id_uint = toUInt128({run_id:UUID})
        ";

        let run_id_str = run_id.to_string();
        let episode_id_str = episode_id.to_string();
        let tags_str = to_map_literal(tags);

        let mut params = HashMap::new();
        params.insert("run_id", run_id_str.as_str());
        params.insert("episode_id", episode_id_str.as_str());
        // Use \\N to indicate NULL; for legacy reasons, stored as `datapoint_name` in the database
        params.insert("datapoint_name", task_name.unwrap_or("\\N"));
        params.insert("tags", tags_str.as_str());
        params.insert("snapshot_hash", &**snapshot_hash);

        self.run_query_synchronous(query.to_string(), &params)
            .await?;
        Ok(())
    }

    async fn get_workflow_evaluation_run_by_episode_id(
        &self,
        episode_id: Uuid,
    ) -> Result<Option<WorkflowEvaluationRunInfo>, Error> {
        let query = r"
        SELECT variant_pins, tags
        FROM DynamicEvaluationRunEpisode
        WHERE episode_id_uint = toUInt128({episode_id:UUID})
        FORMAT JSONEachRow
        ";

        let episode_id_str = episode_id.to_string();
        let params = HashMap::from([("episode_id", episode_id_str.as_str())]);

        let response = self
            .run_query_synchronous(query.to_string(), &params)
            .await?;

        if response.response.is_empty() {
            return Ok(None);
        }

        let info: WorkflowEvaluationRunInfo =
            serde_json::from_str(&response.response).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to deserialize workflow evaluation run info: {e}"),
                })
            })?;

        Ok(Some(info))
    }
}

/// Converts a HashMap to a ClickHouse map literal string.
/// Example: {"key1": "value1", "key2": "value2"} -> "{'key1':'value1','key2':'value2'}"
fn to_map_literal(map: &HashMap<String, String>) -> String {
    let items: Vec<String> = map
        .iter()
        .map(|(k, v)| {
            format!(
                "'{}':'{}'",
                escape_string_for_clickhouse_literal(k),
                escape_string_for_clickhouse_literal(v)
            )
        })
        .collect();
    format!("{{{}}}", items.join(","))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use uuid::Uuid;

    use crate::config::snapshot::SnapshotHash;
    use crate::db::{
        clickhouse::{
            ClickHouseConnectionInfo, ClickHouseResponse, ClickHouseResponseMetadata,
            clickhouse_client::MockClickHouseClient,
            query_builder::test_util::assert_query_contains,
        },
        workflow_evaluation_queries::WorkflowEvaluationQueries,
    };

    #[tokio::test]
    async fn test_list_workflow_evaluation_projects() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "SELECT");
                assert_query_contains(query, "project_name as name");
                assert_query_contains(query, "toUInt32(count()) as count");
                assert_query_contains(query, "FROM DynamicEvaluationRunByProjectName");
                assert_query_contains(query, "GROUP BY project_name");
                assert_query_contains(query, "ORDER BY last_updated DESC");
                assert_query_contains(query, "LIMIT {limit:UInt32}");
                assert_query_contains(query, "OFFSET {offset:UInt32}");

                assert_eq!(params.get("limit"), Some(&"10"));
                assert_eq!(params.get("offset"), Some(&"0"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response:
                        r#"{"name":"project1","count":5,"last_updated":"2025-05-20T16:52:58Z"}"#
                            .to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.list_workflow_evaluation_projects(10, 0).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "project1");
        assert_eq!(result[0].count, 5);
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_projects_with_custom_pagination() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|_query, params| {
                assert_eq!(params.get("limit"), Some(&"50"));
                assert_eq!(params.get("offset"), Some(&"100"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .list_workflow_evaluation_projects(50, 100)
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_projects_multiple_results() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response:
                        r#"{"name":"project1","count":5,"last_updated":"2025-05-20T16:52:58Z"}
{"name":"project2","count":10,"last_updated":"2025-05-20T17:52:58Z"}
{"name":"project3","count":3,"last_updated":"2025-05-20T18:52:58Z"}"#
                            .to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 3,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .list_workflow_evaluation_projects(100, 0)
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].name, "project1");
        assert_eq!(result[1].name, "project2");
        assert_eq!(result[2].name, "project3");
    }

    #[tokio::test]
    async fn test_count_workflow_evaluation_projects() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(
                    query,
                    "
                SELECT toUInt32(countDistinct(project_name)) as count
                FROM DynamicEvaluationRunByProjectName
                WHERE project_name IS NOT NULL
                FORMAT JSONEachRow",
                );
                assert!(params.is_empty());
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":2}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let count = conn.count_workflow_evaluation_projects().await.unwrap();

        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_empty_ids() {
        let mock_clickhouse_client = MockClickHouseClient::new();
        // No expectations set - should not call the database

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.get_workflow_evaluation_runs(&[], None).await.unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_with_ids() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "SELECT");
                assert_query_contains(query, "run_display_name AS name");
                assert_query_contains(query, "uint_to_uuid(run_id_uint) AS id");
                assert_query_contains(query, "variant_pins");
                assert_query_contains(query, "tags");
                assert_query_contains(query, "project_name");
                assert_query_contains(query, "FROM DynamicEvaluationRun");
                assert_query_contains(query, "WHERE run_id_uint IN");
                assert_query_contains(query, "ORDER BY run_id_uint DESC");

                assert!(params.contains_key("run_ids"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"name":"test_run","id":"01968d04-142c-7e53-8ea7-3a3255b518dc","variant_pins":{},"tags":{},"project_name":"test_project","timestamp":"2025-05-01T18:02:56Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let result = conn
            .get_workflow_evaluation_runs(&[run_id], None)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, Some("test_run".to_string()));
        assert_eq!(result[0].id, run_id);
        assert_eq!(result[0].project_name, Some("test_project".to_string()));
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_with_project_name() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "AND project_name = {project_name:String}");
                assert!(params.contains_key("run_ids"));
                assert_eq!(params.get("project_name"), Some(&"my_project"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"name":"filtered_run","id":"01968d04-142c-7e53-8ea7-3a3255b518dc","variant_pins":{},"tags":{},"project_name":"my_project","timestamp":"2025-05-01T18:02:56Z"}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let result = conn
            .get_workflow_evaluation_runs(&[run_id], Some("my_project"))
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].project_name, Some("my_project".to_string()));
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_statistics() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, params| {
                assert_query_contains(query, "SELECT");
                assert_query_contains(query, "FROM FloatMetricFeedbackByTargetId");
                assert_query_contains(query, "FROM BooleanMetricFeedbackByTargetId");
                assert_query_contains(query, "FROM DynamicEvaluationRunEpisodeByRunId");
                assert_query_contains(query, "UNION ALL");
                assert_query_contains(query, "ORDER BY metric_name ASC");
                assert_eq!(
                    params.get("run_id"),
                    Some(&"01968d04-142c-7e53-8ea7-3a3255b518dc")
                );
                true
            })
            .returning(|_, _| {
                // Return mock data matching the expected format
                Ok(ClickHouseResponse {
                    response: r#"{"metric_name":"elapsed_ms","count":49,"avg_metric":91678.72114158163,"stdev":21054.80078125,"is_boolean":false}
{"metric_name":"goated","count":1,"avg_metric":1.0,"stdev":null,"is_boolean":true}
{"metric_name":"solved","count":49,"avg_metric":0.4489795918367347,"stdev":0.5025445456953674,"is_boolean":true}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 3,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_workflow_evaluation_run_statistics(run_id, None)
            .await
            .unwrap();

        assert_eq!(result.len(), 3);

        // Check elapsed_ms (float metric with Wald CI)
        let elapsed_ms = result
            .iter()
            .find(|r| r.metric_name == "elapsed_ms")
            .unwrap();
        assert_eq!(elapsed_ms.count, 49);
        assert!((elapsed_ms.avg_metric - 91678.72114158163).abs() < 0.001);
        assert!((elapsed_ms.stdev.unwrap() - 21054.80078125).abs() < 0.001);
        // Check Wald CI bounds
        let ci_lower = elapsed_ms.ci_lower.unwrap();
        let ci_upper = elapsed_ms.ci_upper.unwrap();
        assert!(
            (ci_lower - 85783.37692283162).abs() < 0.01,
            "ci_lower = {ci_lower}"
        );
        assert!(
            (ci_upper - 97574.06536033163).abs() < 0.01,
            "ci_upper = {ci_upper}"
        );

        // Check goated (boolean metric with Wilson CI)
        let goated = result.iter().find(|r| r.metric_name == "goated").unwrap();
        assert_eq!(goated.count, 1);
        assert!((goated.avg_metric - 1.0).abs() < 0.001);
        assert!(goated.stdev.is_none());
        // Check Wilson CI bounds
        let ci_lower = goated.ci_lower.unwrap();
        let ci_upper = goated.ci_upper.unwrap();
        assert!(
            (ci_lower - 0.20654329147389294).abs() < 0.0001,
            "ci_lower = {ci_lower}"
        );
        assert!((ci_upper - 1.0).abs() < 0.0001, "ci_upper = {ci_upper}");

        // Check solved (boolean metric with Wilson CI)
        let solved = result.iter().find(|r| r.metric_name == "solved").unwrap();
        assert_eq!(solved.count, 49);
        assert!((solved.avg_metric - 0.4489795918367347).abs() < 0.001);
        // Check Wilson CI bounds
        let ci_lower = solved.ci_lower.unwrap();
        let ci_upper = solved.ci_upper.unwrap();
        assert!(
            (ci_lower - 0.31852624929636336).abs() < 0.0001,
            "ci_lower = {ci_lower}"
        );
        assert!(
            (ci_upper - 0.5868513320032188).abs() < 0.0001,
            "ci_upper = {ci_upper}"
        );
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_statistics_with_metric_name_filter() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, params| {
                assert_query_contains(query, "AND metric_name = {metric_name:String}");
                assert_eq!(params.get("metric_name"), Some(&"solved"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"metric_name":"solved","count":49,"avg_metric":0.4489795918367347,"stdev":0.5025445456953674,"is_boolean":true}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_workflow_evaluation_run_statistics(run_id, Some("solved"))
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].metric_name, "solved");
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_run_episodes_by_task_name() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "WITH");
                assert_query_contains(query, "episodes_raw AS");
                assert_query_contains(query, "FROM DynamicEvaluationRunEpisodeByRunId");
                assert_query_contains(query, "group_keys AS");
                assert_query_contains(query, "ifNull(task_name, concat('NULL_EPISODE_'");
                assert_query_contains(query, "feedback_union AS");
                assert_query_contains(query, "FloatMetricFeedbackByTargetId");
                assert_query_contains(query, "BooleanMetricFeedbackByTargetId");
                assert_query_contains(query, "CommentFeedbackByTargetId");
                assert_query_contains(query, "feedback_metric_names");
                assert_query_contains(query, "feedback_values");
                assert_query_contains(query, "LIMIT {limit:UInt32}");
                assert_query_contains(query, "OFFSET {offset:UInt32}");

                assert_eq!(params.get("limit"), Some(&"10"));
                assert_eq!(params.get("offset"), Some(&"0"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"group_key":"task1","episode_id":"01942e26-4693-7e80-8591-47b98e25d721","timestamp":"2025-05-20T16:52:58Z","run_id":"01942e26-4693-7e80-8591-47b98e25d720","tags":{},"task_name":"task1","feedback_metric_names":["metric1"],"feedback_values":["0.5"]}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = uuid::Uuid::parse_str("01942e26-4693-7e80-8591-47b98e25d720").unwrap();
        let result = conn
            .list_workflow_evaluation_run_episodes_by_task_name(&[run_id], 10, 0)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].group_key, "task1");
        assert_eq!(result[0].task_name, Some("task1".to_string()));
        assert_eq!(result[0].feedback_metric_names, vec!["metric1"]);
        assert_eq!(result[0].feedback_values, vec!["0.5"]);
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_run_episodes_by_task_name_empty_run_ids() {
        let mock_clickhouse_client = MockClickHouseClient::new();
        // No expectations set - should not make any queries

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .list_workflow_evaluation_run_episodes_by_task_name(&[], 10, 0)
            .await
            .unwrap();

        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_run_episodes_by_task_name_multiple_results() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"group_key":"task1","episode_id":"01942e26-4693-7e80-8591-47b98e25d721","timestamp":"2025-05-20T16:52:58Z","run_id":"01942e26-4693-7e80-8591-47b98e25d720","tags":{},"task_name":"task1","feedback_metric_names":[],"feedback_values":[]}
{"group_key":"task1","episode_id":"01942e26-4693-7e80-8591-47b98e25d722","timestamp":"2025-05-20T16:53:58Z","run_id":"01942e26-4693-7e80-8591-47b98e25d723","tags":{},"task_name":"task1","feedback_metric_names":[],"feedback_values":[]}
{"group_key":"task2","episode_id":"01942e26-4693-7e80-8591-47b98e25d724","timestamp":"2025-05-20T16:54:58Z","run_id":"01942e26-4693-7e80-8591-47b98e25d720","tags":{},"task_name":"task2","feedback_metric_names":["bool_metric"],"feedback_values":["true"]}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 3,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = uuid::Uuid::parse_str("01942e26-4693-7e80-8591-47b98e25d720").unwrap();
        let result = conn
            .list_workflow_evaluation_run_episodes_by_task_name(&[run_id], 100, 0)
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].group_key, "task1");
        assert_eq!(result[1].group_key, "task1");
        assert_eq!(result[2].group_key, "task2");
    }

    #[tokio::test]
    async fn test_count_workflow_evaluation_run_episodes_by_task_name() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "WITH");
                assert_query_contains(query, "episodes_raw AS");
                assert_query_contains(query, "FROM DynamicEvaluationRunEpisodeByRunId");
                assert_query_contains(
                    query,
                    "countDistinct(ifNull(task_name, concat('NULL_EPISODE_'",
                );
                assert!(params.is_empty());
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":5}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = uuid::Uuid::parse_str("01942e26-4693-7e80-8591-47b98e25d720").unwrap();
        let count = conn
            .count_workflow_evaluation_run_episodes_by_task_name(&[run_id])
            .await
            .unwrap();

        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn test_count_workflow_evaluation_run_episodes_by_task_name_empty_run_ids() {
        let mock_clickhouse_client = MockClickHouseClient::new();
        // No expectations set - should not make any queries

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let count = conn
            .count_workflow_evaluation_run_episodes_by_task_name(&[])
            .await
            .unwrap();

        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_episodes_with_feedback() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "SELECT");
                assert_query_contains(query, "uint_to_uuid(e.episode_id_uint) AS episode_id");
                assert_query_contains(query, "FROM DynamicEvaluationRunEpisodeByRunId");
                assert_query_contains(query, "WHERE toUInt128(toUUID({run_id:String})) = run_id_uint");
                assert_query_contains(query, "ORDER BY episode_id_uint DESC");
                assert_query_contains(query, "LIMIT {limit:UInt32}");
                assert_query_contains(query, "OFFSET {offset:UInt32}");
                assert_query_contains(query, "FloatMetricFeedbackByTargetId");
                assert_query_contains(query, "BooleanMetricFeedbackByTargetId");
                assert_query_contains(query, "CommentFeedbackByTargetId");
                assert_query_contains(query, "feedback_metric_names");
                assert_query_contains(query, "feedback_values");

                assert_eq!(params.get("run_id"), Some(&"01968d04-142c-7e53-8ea7-3a3255b518dc"));
                assert_eq!(params.get("limit"), Some(&"10"));
                assert_eq!(params.get("offset"), Some(&"0"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"episode_id":"0aaedb76-b457-7eae-aa62-145b73aa3e24","timestamp":"2025-05-01T18:02:56Z","run_id":"01968d04-142c-7e53-8ea7-3a3255b518dc","tags":{"foo":"bar"},"task_name":null,"feedback_metric_names":["elapsed_ms","solved"],"feedback_values":["105946.19","false"]}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let result = conn
            .get_workflow_evaluation_run_episodes_with_feedback(run_id, 10, 0)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].episode_id,
            Uuid::parse_str("0aaedb76-b457-7eae-aa62-145b73aa3e24").unwrap()
        );
        assert_eq!(result[0].run_id, run_id);
        assert_eq!(result[0].task_name, None);
        assert_eq!(
            result[0].feedback_metric_names,
            vec!["elapsed_ms", "solved"]
        );
        assert_eq!(result[0].feedback_values, vec!["105946.19", "false"]);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_episodes_with_feedback_pagination() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|_query, params| {
                assert_eq!(params.get("limit"), Some(&"5"));
                assert_eq!(params.get("offset"), Some(&"10"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let result = conn
            .get_workflow_evaluation_run_episodes_with_feedback(run_id, 5, 10)
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_episodes_with_feedback_multiple() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"episode_id":"0aaedb76-b457-7eae-aa62-145b73aa3e24","timestamp":"2025-05-01T18:02:56Z","run_id":"01968d04-142c-7e53-8ea7-3a3255b518dc","tags":{},"task_name":null,"feedback_metric_names":["metric1"],"feedback_values":["value1"]}
{"episode_id":"0aaedb76-b457-7c86-8a61-2ffa01519447","timestamp":"2025-05-01T18:02:57Z","run_id":"01968d04-142c-7e53-8ea7-3a3255b518dc","tags":{},"task_name":"task1","feedback_metric_names":["metric1","metric2"],"feedback_values":["val1","val2"]}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let result = conn
            .get_workflow_evaluation_run_episodes_with_feedback(run_id, 10, 0)
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].task_name, None);
        assert_eq!(result[1].task_name, Some("task1".to_string()));
        assert_eq!(result[1].feedback_metric_names, vec!["metric1", "metric2"]);
    }

    #[tokio::test]
    async fn test_count_workflow_evaluation_run_episodes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "SELECT toUInt32(count()) as count");
                assert_query_contains(query, "FROM DynamicEvaluationRunEpisodeByRunId");
                assert_query_contains(
                    query,
                    "WHERE toUInt128(toUUID({run_id:String})) = run_id_uint",
                );
                assert_eq!(
                    params.get("run_id"),
                    Some(&"01968d04-142c-7e53-8ea7-3a3255b518dc")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":50}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let count = conn
            .count_workflow_evaluation_run_episodes(run_id)
            .await
            .unwrap();

        assert_eq!(count, 50);
    }

    #[tokio::test]
    async fn test_count_workflow_evaluation_run_episodes_empty() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"count":0}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let count = conn
            .count_workflow_evaluation_run_episodes(run_id)
            .await
            .unwrap();

        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_insert_workflow_evaluation_run() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "INSERT INTO DynamicEvaluationRun");
                assert_query_contains(query, "run_id_uint");
                assert_query_contains(query, "variant_pins");
                assert_query_contains(query, "tags");
                assert_query_contains(query, "project_name");
                assert_query_contains(query, "run_display_name");
                assert_query_contains(query, "snapshot_hash");
                assert_query_contains(query, "toUInt128({run_id:UUID})");
                assert_query_contains(query, "{variant_pins:Map(String, String)}");
                assert_query_contains(query, "{tags:Map(String, String)}");

                assert!(params.contains_key("run_id"));
                assert!(params.contains_key("variant_pins"));
                assert!(params.contains_key("tags"));
                assert!(params.contains_key("project_name"));
                assert!(params.contains_key("run_display_name"));
                assert!(params.contains_key("snapshot_hash"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 1,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::now_v7();
        let variant_pins = HashMap::from([("fn1".to_string(), "var1".to_string())]);
        let tags = HashMap::from([("key".to_string(), "value".to_string())]);
        let snapshot_hash = SnapshotHash::default();

        let result = conn
            .insert_workflow_evaluation_run(
                run_id,
                &variant_pins,
                &tags,
                Some("my_project"),
                Some("My Run"),
                &snapshot_hash,
            )
            .await;

        assert!(
            result.is_ok(),
            "insert_workflow_evaluation_run should succeed"
        );
    }

    #[tokio::test]
    async fn test_insert_workflow_evaluation_run_with_nulls() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|_query, params| {
                // When project_name and display_name are None, they should be "\\N"
                assert_eq!(params.get("project_name"), Some(&"\\N"));
                assert_eq!(params.get("run_display_name"), Some(&"\\N"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 1,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::now_v7();
        let variant_pins = HashMap::new();
        let tags = HashMap::new();
        let snapshot_hash = SnapshotHash::default();

        let result = conn
            .insert_workflow_evaluation_run(
                run_id,
                &variant_pins,
                &tags,
                None,
                None,
                &snapshot_hash,
            )
            .await;

        assert!(
            result.is_ok(),
            "insert_workflow_evaluation_run with nulls should succeed"
        );
    }

    #[tokio::test]
    async fn test_insert_workflow_evaluation_run_episode() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "INSERT INTO DynamicEvaluationRunEpisode");
                assert_query_contains(query, "run_id");
                assert_query_contains(query, "episode_id_uint");
                assert_query_contains(query, "variant_pins");
                assert_query_contains(query, "datapoint_name");
                assert_query_contains(query, "tags");
                assert_query_contains(query, "snapshot_hash");
                assert_query_contains(query, "SELECT");
                assert_query_contains(query, "FROM DynamicEvaluationRun");
                assert_query_contains(query, "mapUpdate(tags, {tags:Map(String, String)})");

                assert!(params.contains_key("run_id"));
                assert!(params.contains_key("episode_id"));
                assert!(params.contains_key("datapoint_name"));
                assert!(params.contains_key("tags"));
                assert!(params.contains_key("snapshot_hash"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 1,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let tags = HashMap::from([("episode_tag".to_string(), "value".to_string())]);
        let snapshot_hash = SnapshotHash::default();

        let result = conn
            .insert_workflow_evaluation_run_episode(
                run_id,
                episode_id,
                Some("my_task"),
                &tags,
                &snapshot_hash,
            )
            .await;

        assert!(
            result.is_ok(),
            "insert_workflow_evaluation_run_episode should succeed"
        );
    }

    #[tokio::test]
    async fn test_insert_workflow_evaluation_run_episode_with_null_task_name() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|_query, params| {
                // When task_name is None, datapoint_name should be "\\N"
                assert_eq!(params.get("datapoint_name"), Some(&"\\N"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 1,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let run_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let tags = HashMap::new();
        let snapshot_hash = SnapshotHash::default();

        let result = conn
            .insert_workflow_evaluation_run_episode(run_id, episode_id, None, &tags, &snapshot_hash)
            .await;

        assert!(
            result.is_ok(),
            "insert_workflow_evaluation_run_episode with null task_name should succeed"
        );
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_by_episode_id() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let episode_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, params| {
                assert_query_contains(query, "SELECT variant_pins, tags");
                assert_query_contains(query, "FROM DynamicEvaluationRunEpisode");
                assert_query_contains(
                    query,
                    "WHERE episode_id_uint = toUInt128({episode_id:UUID})",
                );
                assert_query_contains(query, "FORMAT JSONEachRow");

                assert_eq!(
                    params.get("episode_id"),
                    Some(&"01968d04-142c-7e53-8ea7-3a3255b518dc")
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"variant_pins":{"fn1":"var1"},"tags":{"key":"value"}}"#
                        .to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_workflow_evaluation_run_by_episode_id(episode_id)
            .await
            .unwrap();

        assert!(result.is_some(), "should return Some for existing episode");
        let info = result.unwrap();
        assert_eq!(
            info.variant_pins.get("fn1"),
            Some(&"var1".to_string()),
            "variant_pins should contain fn1 -> var1"
        );
        assert_eq!(
            info.tags.get("key"),
            Some(&"value".to_string()),
            "tags should contain key -> value"
        );
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_by_episode_id_not_found() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let episode_id = Uuid::now_v7();
        let result = conn
            .get_workflow_evaluation_run_by_episode_id(episode_id)
            .await
            .unwrap();

        assert!(
            result.is_none(),
            "should return None for non-existent episode"
        );
    }
}
