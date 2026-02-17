//! Workflow evaluation queries for Postgres.
//!
//! This module implements both read and write operations for workflow evaluation tables in Postgres.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::PgRow;
use sqlx::{PgPool, QueryBuilder, Row};
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::db::query_helpers::uuid_to_datetime;
use crate::db::workflow_evaluation_queries::{
    GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow, WorkflowEvaluationProjectRow,
    WorkflowEvaluationQueries, WorkflowEvaluationRunEpisodeWithFeedbackRow,
    WorkflowEvaluationRunInfo, WorkflowEvaluationRunRow, WorkflowEvaluationRunStatisticsRaw,
    WorkflowEvaluationRunStatisticsRow, WorkflowEvaluationRunWithEpisodeCountRow,
};
use crate::error::{Error, ErrorDetails};
use crate::statistics_util::{wald_confint, wilson_confint};

use super::PostgresConnectionInfo;

#[derive(sqlx::FromRow)]
struct WorkflowEvaluationRunByEpisodeRow {
    variant_pins: serde_json::Value,
    tags: serde_json::Value,
}

// =====================================================================
// WorkflowEvaluationQueries trait implementation
// =====================================================================

#[async_trait]
impl WorkflowEvaluationQueries for PostgresConnectionInfo {
    async fn list_workflow_evaluation_projects(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<WorkflowEvaluationProjectRow>, Error> {
        let pool = self.get_pool_result()?;
        let mut qb = build_list_workflow_evaluation_projects_query(limit, offset);

        let rows: Vec<WorkflowEvaluationProjectRow> = qb.build_query_as().fetch_all(pool).await?;

        Ok(rows)
    }

    async fn count_workflow_evaluation_projects(&self) -> Result<u32, Error> {
        let pool = self.get_pool_result()?;

        let count: i32 = sqlx::query_scalar!(
            r#"
            SELECT COUNT(DISTINCT project_name)::INT as "count!"
            FROM tensorzero.workflow_evaluation_runs
            WHERE project_name IS NOT NULL AND staled_at IS NULL
            "#
        )
        .fetch_one(pool)
        .await?;

        Ok(count as u32)
    }

    async fn search_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        project_name: Option<&str>,
        search_query: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error> {
        let pool = self.get_pool_result()?;
        let mut qb =
            build_search_workflow_evaluation_runs_query(limit, offset, project_name, search_query);

        let rows: Vec<WorkflowEvaluationRunRow> = qb.build_query_as().fetch_all(pool).await?;

        Ok(rows)
    }

    async fn list_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        run_id: Option<Uuid>,
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunWithEpisodeCountRow>, Error> {
        let pool = self.get_pool_result()?;
        let mut qb = build_list_workflow_evaluation_runs_query(limit, offset, run_id, project_name);

        let rows: Vec<WorkflowEvaluationRunWithEpisodeCountRow> =
            qb.build_query_as().fetch_all(pool).await?;

        Ok(rows)
    }

    async fn count_workflow_evaluation_runs(&self) -> Result<u32, Error> {
        let pool = self.get_pool_result()?;

        let count: i32 = sqlx::query_scalar!(
            r#"
            SELECT COUNT(*)::INT as "count!"
            FROM tensorzero.workflow_evaluation_runs
            WHERE staled_at IS NULL
            "#
        )
        .fetch_one(pool)
        .await?;

        Ok(count as u32)
    }

    async fn get_workflow_evaluation_runs(
        &self,
        run_ids: &[Uuid],
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error> {
        if run_ids.is_empty() {
            return Ok(vec![]);
        }

        let pool = self.get_pool_result()?;
        let mut qb = build_get_workflow_evaluation_runs_query(run_ids, project_name);

        let rows: Vec<WorkflowEvaluationRunRow> = qb.build_query_as().fetch_all(pool).await?;

        Ok(rows)
    }

    async fn get_workflow_evaluation_run_statistics(
        &self,
        run_id: Uuid,
        metric_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunStatisticsRow>, Error> {
        let pool = self.get_pool_result()?;
        let raw_stats =
            get_workflow_evaluation_run_statistics_raw(pool, run_id, metric_name).await?;

        // Compute confidence intervals in Rust (same as ClickHouse implementation)
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

        let pool = self.get_pool_result()?;
        list_workflow_evaluation_run_episodes_by_task_name_impl(pool, run_ids, limit, offset).await
    }

    async fn count_workflow_evaluation_run_episodes_by_task_name(
        &self,
        run_ids: &[Uuid],
    ) -> Result<u32, Error> {
        if run_ids.is_empty() {
            return Ok(0);
        }

        let pool = self.get_pool_result()?;
        let mut qb = build_count_workflow_evaluation_run_episodes_by_task_name_query(run_ids);

        let row: PgRow = qb.build().fetch_one(pool).await?;
        let count: i32 = row.get("count");

        Ok(count as u32)
    }

    async fn get_workflow_evaluation_run_episodes_with_feedback(
        &self,
        run_id: Uuid,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<WorkflowEvaluationRunEpisodeWithFeedbackRow>, Error> {
        let pool = self.get_pool_result()?;
        get_workflow_evaluation_run_episodes_with_feedback_impl(pool, run_id, limit, offset).await
    }

    async fn count_workflow_evaluation_run_episodes(&self, run_id: Uuid) -> Result<u32, Error> {
        let pool = self.get_pool_result()?;
        let mut qb = build_count_workflow_evaluation_run_episodes_query(run_id);

        let row: PgRow = qb.build().fetch_one(pool).await?;
        let count: i32 = row.get("count");

        Ok(count as u32)
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
        let pool = self.get_pool_result()?;

        let variant_pins_json = serde_json::to_value(variant_pins).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize variant_pins: {e}"),
            })
        })?;
        let tags_json = serde_json::to_value(tags).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize tags: {e}"),
            })
        })?;
        let created_at = uuid_to_datetime(run_id)?;

        let mut qb = build_insert_workflow_evaluation_run_query(
            run_id,
            &variant_pins_json,
            &tags_json,
            project_name,
            run_display_name,
            snapshot_hash.as_bytes(),
            created_at,
        );

        qb.build().execute(pool).await.map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to insert workflow evaluation run: {e}"),
            })
        })?;

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
        let pool = self.get_pool_result()?;

        let tags_json = serde_json::to_value(tags).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize tags: {e}"),
            })
        })?;
        let created_at = uuid_to_datetime(episode_id)?;

        let mut qb = build_insert_workflow_evaluation_run_episode_query(
            episode_id,
            run_id,
            task_name,
            &tags_json,
            snapshot_hash.as_bytes(),
            created_at,
        );

        qb.build().execute(pool).await.map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to insert workflow evaluation run episode: {e}"),
            })
        })?;

        Ok(())
    }

    async fn get_workflow_evaluation_run_by_episode_id(
        &self,
        episode_id: Uuid,
    ) -> Result<Option<WorkflowEvaluationRunInfo>, Error> {
        let pool = self.get_pool_result()?;
        let mut qb = build_get_workflow_evaluation_run_by_episode_id_query(episode_id);

        let row: Option<WorkflowEvaluationRunByEpisodeRow> =
            qb.build_query_as().fetch_optional(pool).await?;

        match row {
            Some(row) => {
                let variant_pins: HashMap<String, String> =
                    serde_json::from_value(row.variant_pins).unwrap_or_default();
                let tags: HashMap<String, String> =
                    serde_json::from_value(row.tags).unwrap_or_default();
                Ok(Some(WorkflowEvaluationRunInfo { variant_pins, tags }))
            }
            None => Ok(None),
        }
    }
}

// =====================================================================
// Query builder functions (for unit testing)
// =====================================================================

/// Builds a query to list workflow evaluation projects.
fn build_list_workflow_evaluation_projects_query(
    limit: u32,
    offset: u32,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            project_name as name,
            COUNT(*)::INT as count,
            MAX(updated_at) as last_updated
        FROM tensorzero.workflow_evaluation_runs
        WHERE project_name IS NOT NULL AND staled_at IS NULL
        GROUP BY project_name
        ORDER BY MAX(updated_at) DESC
        LIMIT ",
    );
    qb.push_bind(limit as i64);
    qb.push(" OFFSET ");
    qb.push_bind(offset as i64);

    qb
}

/// Builds a query to search workflow evaluation runs.
fn build_search_workflow_evaluation_runs_query(
    limit: u32,
    offset: u32,
    project_name: Option<&str>,
    search_query: Option<&str>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            run_display_name as name,
            run_id as id,
            variant_pins,
            tags,
            project_name,
            created_at as timestamp
        FROM tensorzero.workflow_evaluation_runs
        WHERE staled_at IS NULL
        ",
    );

    if let Some(project_name) = project_name {
        qb.push(" AND project_name = ");
        qb.push_bind(project_name.to_string());
    }

    if let Some(search_query) = search_query {
        qb.push(" AND (run_display_name ILIKE ");
        qb.push_bind(format!("%{search_query}%"));
        qb.push(" OR run_id::TEXT ILIKE ");
        qb.push_bind(format!("%{search_query}%"));
        qb.push(")");
    }

    qb.push(" ORDER BY created_at DESC LIMIT ");
    qb.push_bind(limit as i64);
    qb.push(" OFFSET ");
    qb.push_bind(offset as i64);

    qb
}

/// Builds a query to list workflow evaluation runs with episode counts.
fn build_list_workflow_evaluation_runs_query(
    limit: u32,
    offset: u32,
    run_id: Option<Uuid>,
    project_name: Option<&str>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        WITH filtered_runs AS (
            SELECT
                run_display_name as name,
                run_id as id,
                variant_pins,
                tags,
                project_name,
                created_at as timestamp
            FROM tensorzero.workflow_evaluation_runs
            WHERE staled_at IS NULL
        ",
    );

    if let Some(run_id) = run_id {
        qb.push(" AND run_id = ");
        qb.push_bind(run_id);
    } else if let Some(project_name) = project_name {
        qb.push(" AND project_name = ");
        qb.push_bind(project_name.to_string());
    }

    qb.push(
        r"
            ORDER BY id DESC
            LIMIT ",
    );
    qb.push_bind(limit as i64);
    qb.push(" OFFSET ");
    qb.push_bind(offset as i64);
    qb.push(
        r"
        ),
        episode_counts AS (
            SELECT
                run_id,
                COUNT(*)::INT as num_episodes
            FROM tensorzero.workflow_evaluation_run_episodes
            WHERE run_id IN (SELECT id FROM filtered_runs) AND staled_at IS NULL
            GROUP BY run_id
        )
        SELECT
            r.name,
            r.id,
            r.variant_pins,
            r.tags,
            r.project_name,
            COALESCE(e.num_episodes, 0)::INT as num_episodes,
            r.timestamp
        FROM filtered_runs r
        LEFT JOIN episode_counts e ON r.id = e.run_id
        ORDER BY r.id DESC
        ",
    );

    qb
}

/// Builds a query to get workflow evaluation runs by IDs.
fn build_get_workflow_evaluation_runs_query(
    run_ids: &[Uuid],
    project_name: Option<&str>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            run_display_name as name,
            run_id as id,
            variant_pins,
            tags,
            project_name,
            created_at as timestamp
        FROM tensorzero.workflow_evaluation_runs
        WHERE staled_at IS NULL AND run_id = ANY(",
    );
    qb.push_bind(run_ids.to_vec());
    qb.push(")");

    if let Some(project_name) = project_name {
        qb.push(" AND project_name = ");
        qb.push_bind(project_name.to_string());
    }

    qb.push(" ORDER BY run_id DESC");

    qb
}

/// Builds a query to get raw statistics for a workflow evaluation run.
fn build_get_workflow_evaluation_run_statistics_query(
    run_id: Uuid,
    metric_name: Option<&str>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        WITH episodes AS (
            SELECT episode_id
            FROM tensorzero.workflow_evaluation_run_episodes
            WHERE run_id = ",
    );
    qb.push_bind(run_id);
    qb.push(
        r" AND staled_at IS NULL
        ),
        float_stats AS (
            SELECT
                metric_name,
                COUNT(*)::INT as count,
                AVG(value)::DOUBLE PRECISION as avg_metric,
                STDDEV_SAMP(value)::DOUBLE PRECISION as stdev,
                false as is_boolean
            FROM tensorzero.float_metric_feedback
            WHERE target_id IN (SELECT episode_id FROM episodes)
        ",
    );

    if let Some(metric_name) = metric_name {
        qb.push(" AND metric_name = ");
        qb.push_bind(metric_name.to_string());
    }

    qb.push(
        r"
            GROUP BY metric_name
        ),
        boolean_stats AS (
            SELECT
                metric_name,
                COUNT(*)::INT as count,
                AVG(value::INT)::DOUBLE PRECISION as avg_metric,
                STDDEV_SAMP(value::INT)::DOUBLE PRECISION as stdev,
                true as is_boolean
            FROM tensorzero.boolean_metric_feedback
            WHERE target_id IN (SELECT episode_id FROM episodes)
        ",
    );

    if let Some(metric_name) = metric_name {
        qb.push(" AND metric_name = ");
        qb.push_bind(metric_name.to_string());
    }

    qb.push(
        r"
            GROUP BY metric_name
        )
        SELECT * FROM float_stats
        UNION ALL
        SELECT * FROM boolean_stats
        ORDER BY metric_name ASC
        ",
    );

    qb
}

/// Builds a query to count distinct task names for workflow evaluation run episodes.
fn build_count_workflow_evaluation_run_episodes_by_task_name_query(
    run_ids: &[Uuid],
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT COUNT(DISTINCT COALESCE(task_name, 'NULL_EPISODE_' || episode_id::TEXT))::INT as count
        FROM tensorzero.workflow_evaluation_run_episodes
        WHERE run_id = ANY(",
    );
    qb.push_bind(run_ids.to_vec());
    qb.push(") AND staled_at IS NULL");

    qb
}

/// Builds a query to list workflow evaluation run episodes grouped by task name.
fn build_list_workflow_evaluation_run_episodes_by_task_name_query(
    run_ids: &[Uuid],
    limit: u32,
    offset: u32,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        WITH episodes_with_key AS (
            SELECT
                episode_id,
                run_id,
                tags,
                updated_at,
                task_name,
                COALESCE(task_name, 'NULL_EPISODE_' || episode_id::TEXT) as group_key
            FROM tensorzero.workflow_evaluation_run_episodes
            WHERE run_id = ANY(",
    );
    qb.push_bind(run_ids.to_vec());
    qb.push(
        r") AND staled_at IS NULL
        ),
        group_keys AS (
            SELECT
                group_key,
                MAX(updated_at) as last_updated
            FROM episodes_with_key
            GROUP BY group_key
            ORDER BY last_updated DESC
            LIMIT ",
    );
    qb.push_bind(limit as i64);
    qb.push(" OFFSET ");
    qb.push_bind(offset as i64);
    qb.push(
        r"
        ),
        episodes AS (
            SELECT e.*
            FROM episodes_with_key e
            WHERE e.group_key IN (SELECT group_key FROM group_keys)
        ),
        feedback_union AS (
            SELECT * FROM (
                SELECT DISTINCT ON (target_id, metric_name)
                    target_id, metric_name, value::TEXT as value
                FROM tensorzero.float_metric_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
                ORDER BY target_id, metric_name, created_at DESC
            ) float_fb
            UNION ALL
            SELECT * FROM (
                SELECT DISTINCT ON (target_id, metric_name)
                    target_id, metric_name, value::INT::TEXT as value
                FROM tensorzero.boolean_metric_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
                ORDER BY target_id, metric_name, created_at DESC
            ) bool_fb
            UNION ALL
            SELECT target_id, 'comment' as metric_name, value
            FROM tensorzero.comment_feedback
            WHERE target_id IN (SELECT episode_id FROM episodes)
        )
        SELECT
            e.group_key,
            e.episode_id,
            e.updated_at as timestamp,
            e.run_id,
            e.tags,
            e.task_name,
            COALESCE(ARRAY_AGG(f.metric_name ORDER BY f.metric_name) FILTER (WHERE f.metric_name IS NOT NULL), ARRAY[]::TEXT[]) as feedback_metric_names,
            COALESCE(ARRAY_AGG(f.value ORDER BY f.metric_name) FILTER (WHERE f.metric_name IS NOT NULL), ARRAY[]::TEXT[]) as feedback_values
        FROM episodes e
        JOIN group_keys g ON e.group_key = g.group_key
        LEFT JOIN feedback_union f ON f.target_id = e.episode_id
        GROUP BY e.group_key, e.episode_id, e.run_id, e.tags, e.task_name, e.updated_at, g.last_updated
        ORDER BY g.last_updated DESC, e.group_key, e.episode_id
        ",
    );

    qb
}

/// Builds a query to get workflow evaluation run episodes with their feedback.
fn build_get_workflow_evaluation_run_episodes_with_feedback_query(
    run_id: Uuid,
    limit: u32,
    offset: u32,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        WITH episodes AS (
            SELECT
                episode_id,
                run_id,
                tags,
                updated_at,
                task_name
            FROM tensorzero.workflow_evaluation_run_episodes
            WHERE run_id = ",
    );
    qb.push_bind(run_id);
    qb.push(
        r" AND staled_at IS NULL
            ORDER BY episode_id DESC
            LIMIT ",
    );
    qb.push_bind(limit as i64);
    qb.push(" OFFSET ");
    qb.push_bind(offset as i64);
    qb.push(
        r"
        ),
        feedback_union AS (
            SELECT * FROM (
                SELECT DISTINCT ON (target_id, metric_name)
                    target_id, metric_name, value::TEXT as value
                FROM tensorzero.float_metric_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
                ORDER BY target_id, metric_name, created_at DESC
            ) float_fb
            UNION ALL
            SELECT * FROM (
                SELECT DISTINCT ON (target_id, metric_name)
                    target_id, metric_name, value::INT::TEXT as value
                FROM tensorzero.boolean_metric_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
                ORDER BY target_id, metric_name, created_at DESC
            ) bool_fb
            UNION ALL
            SELECT target_id, 'comment' as metric_name, value
            FROM tensorzero.comment_feedback
            WHERE target_id IN (SELECT episode_id FROM episodes)
        )
        SELECT
            e.episode_id,
            e.updated_at as timestamp,
            e.run_id,
            e.tags,
            e.task_name,
            COALESCE(ARRAY_AGG(f.metric_name ORDER BY f.metric_name) FILTER (WHERE f.metric_name IS NOT NULL), ARRAY[]::TEXT[]) as feedback_metric_names,
            COALESCE(ARRAY_AGG(f.value ORDER BY f.metric_name) FILTER (WHERE f.metric_name IS NOT NULL), ARRAY[]::TEXT[]) as feedback_values
        FROM episodes e
        LEFT JOIN feedback_union f ON f.target_id = e.episode_id
        GROUP BY e.episode_id, e.run_id, e.tags, e.task_name, e.updated_at
        ORDER BY e.episode_id DESC
        ",
    );

    qb
}

/// Builds a query to count workflow evaluation run episodes.
fn build_count_workflow_evaluation_run_episodes_query(
    run_id: Uuid,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT COUNT(*)::INT as count
        FROM tensorzero.workflow_evaluation_run_episodes
        WHERE run_id = ",
    );
    qb.push_bind(run_id);
    qb.push(" AND staled_at IS NULL");

    qb
}

/// Builds a query to insert a workflow evaluation run.
fn build_insert_workflow_evaluation_run_query(
    run_id: Uuid,
    variant_pins_json: &serde_json::Value,
    tags_json: &serde_json::Value,
    project_name: Option<&str>,
    run_display_name: Option<&str>,
    snapshot_hash_bytes: &[u8],
    created_at: DateTime<Utc>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.workflow_evaluation_runs
            (run_id, variant_pins, tags, project_name, run_display_name, snapshot_hash, created_at)
        VALUES (",
    );
    qb.push_bind(run_id);
    qb.push(", ");
    qb.push_bind(variant_pins_json.clone());
    qb.push(", ");
    qb.push_bind(tags_json.clone());
    qb.push(", ");
    qb.push_bind(project_name.map(|s| s.to_string()));
    qb.push(", ");
    qb.push_bind(run_display_name.map(|s| s.to_string()));
    qb.push(", ");
    qb.push_bind(snapshot_hash_bytes.to_vec());
    qb.push(", ");
    qb.push_bind(created_at);
    qb.push(")");

    qb
}

/// Builds a query to insert a workflow evaluation run episode.
fn build_insert_workflow_evaluation_run_episode_query(
    episode_id: Uuid,
    run_id: Uuid,
    task_name: Option<&str>,
    tags_json: &serde_json::Value,
    snapshot_hash_bytes: &[u8],
    created_at: DateTime<Utc>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.workflow_evaluation_run_episodes
            (episode_id, run_id, variant_pins, task_name, tags, snapshot_hash, created_at)
        SELECT
            ",
    );
    qb.push_bind(episode_id);
    qb.push(", ");
    qb.push_bind(run_id);
    qb.push(
        r",
            r.variant_pins,
            ",
    );
    qb.push_bind(task_name.map(|s| s.to_string()));
    qb.push(", r.tags || ");
    qb.push_bind(tags_json.clone());
    qb.push(", ");
    qb.push_bind(snapshot_hash_bytes.to_vec());
    qb.push(", ");
    qb.push_bind(created_at);
    qb.push(
        r"
        FROM tensorzero.workflow_evaluation_runs r
        WHERE r.run_id = ",
    );
    qb.push_bind(run_id);

    qb
}

/// Builds a query to get a workflow evaluation run by episode ID.
fn build_get_workflow_evaluation_run_by_episode_id_query(
    episode_id: Uuid,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT variant_pins, tags
        FROM tensorzero.workflow_evaluation_run_episodes
        WHERE episode_id = ",
    );
    qb.push_bind(episode_id);
    qb.push(" AND staled_at IS NULL");

    qb
}

// =====================================================================
// Helper functions for executing queries
// =====================================================================

/// Helper function to get raw statistics for a workflow evaluation run.
async fn get_workflow_evaluation_run_statistics_raw(
    pool: &PgPool,
    run_id: Uuid,
    metric_name: Option<&str>,
) -> Result<Vec<WorkflowEvaluationRunStatisticsRaw>, Error> {
    let mut qb = build_get_workflow_evaluation_run_statistics_query(run_id, metric_name);

    let rows: Vec<WorkflowEvaluationRunStatisticsRaw> = qb.build_query_as().fetch_all(pool).await?;

    Ok(rows)
}

/// Helper function to list workflow evaluation run episodes grouped by task name.
async fn list_workflow_evaluation_run_episodes_by_task_name_impl(
    pool: &PgPool,
    run_ids: &[Uuid],
    limit: u32,
    offset: u32,
) -> Result<Vec<GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow>, Error> {
    let mut qb =
        build_list_workflow_evaluation_run_episodes_by_task_name_query(run_ids, limit, offset);

    let rows: Vec<GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow> =
        qb.build_query_as().fetch_all(pool).await?;

    Ok(rows)
}

/// Helper function to get workflow evaluation run episodes with their feedback.
async fn get_workflow_evaluation_run_episodes_with_feedback_impl(
    pool: &PgPool,
    run_id: Uuid,
    limit: u32,
    offset: u32,
) -> Result<Vec<WorkflowEvaluationRunEpisodeWithFeedbackRow>, Error> {
    let mut qb =
        build_get_workflow_evaluation_run_episodes_with_feedback_query(run_id, limit, offset);

    let rows: Vec<WorkflowEvaluationRunEpisodeWithFeedbackRow> =
        qb.build_query_as().fetch_all(pool).await?;

    Ok(rows)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_helpers::assert_query_equals;

    #[test]
    fn test_build_list_workflow_evaluation_projects_query() {
        let qb = build_list_workflow_evaluation_projects_query(10, 5);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                project_name as name,
                COUNT(*)::INT as count,
                MAX(updated_at) as last_updated
            FROM tensorzero.workflow_evaluation_runs
            WHERE project_name IS NOT NULL AND staled_at IS NULL
            GROUP BY project_name
            ORDER BY MAX(updated_at) DESC
            LIMIT $1 OFFSET $2
            ",
        );
    }

    #[test]
    fn test_build_search_workflow_evaluation_runs_query_no_filters() {
        let qb = build_search_workflow_evaluation_runs_query(10, 0, None, None);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_display_name as name,
                run_id as id,
                variant_pins,
                tags,
                project_name,
                created_at as timestamp
            FROM tensorzero.workflow_evaluation_runs
            WHERE staled_at IS NULL
            ORDER BY created_at DESC LIMIT $1 OFFSET $2
            ",
        );
    }

    #[test]
    fn test_build_search_workflow_evaluation_runs_query_with_project_name() {
        let qb = build_search_workflow_evaluation_runs_query(10, 0, Some("my_project"), None);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_display_name as name,
                run_id as id,
                variant_pins,
                tags,
                project_name,
                created_at as timestamp
            FROM tensorzero.workflow_evaluation_runs
            WHERE staled_at IS NULL
            AND project_name = $1
            ORDER BY created_at DESC LIMIT $2 OFFSET $3
            ",
        );
    }

    #[test]
    fn test_build_search_workflow_evaluation_runs_query_with_search() {
        let qb = build_search_workflow_evaluation_runs_query(10, 0, None, Some("test"));
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_display_name as name,
                run_id as id,
                variant_pins,
                tags,
                project_name,
                created_at as timestamp
            FROM tensorzero.workflow_evaluation_runs
            WHERE staled_at IS NULL
            AND (run_display_name ILIKE $1 OR run_id::TEXT ILIKE $2)
            ORDER BY created_at DESC LIMIT $3 OFFSET $4
            ",
        );
    }

    #[test]
    fn test_build_list_workflow_evaluation_runs_query_no_filters() {
        let qb = build_list_workflow_evaluation_runs_query(10, 0, None, None);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            WITH filtered_runs AS (
                SELECT
                    run_display_name as name,
                    run_id as id,
                    variant_pins,
                    tags,
                    project_name,
                    created_at as timestamp
                FROM tensorzero.workflow_evaluation_runs
                WHERE staled_at IS NULL
                ORDER BY id DESC
                LIMIT $1 OFFSET $2
            ),
            episode_counts AS (
                SELECT
                    run_id,
                    COUNT(*)::INT as num_episodes
                FROM tensorzero.workflow_evaluation_run_episodes
                WHERE run_id IN (SELECT id FROM filtered_runs) AND staled_at IS NULL
                GROUP BY run_id
            )
            SELECT
                r.name,
                r.id,
                r.variant_pins,
                r.tags,
                r.project_name,
                COALESCE(e.num_episodes, 0)::INT as num_episodes,
                r.timestamp
            FROM filtered_runs r
            LEFT JOIN episode_counts e ON r.id = e.run_id
            ORDER BY r.id DESC
            ",
        );
    }

    #[test]
    fn test_build_list_workflow_evaluation_runs_query_with_run_id() {
        let run_id = Uuid::now_v7();
        let qb = build_list_workflow_evaluation_runs_query(10, 0, Some(run_id), None);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            WITH filtered_runs AS (
                SELECT
                    run_display_name as name,
                    run_id as id,
                    variant_pins,
                    tags,
                    project_name,
                    created_at as timestamp
                FROM tensorzero.workflow_evaluation_runs
                WHERE staled_at IS NULL
                AND run_id = $1
                ORDER BY id DESC
                LIMIT $2 OFFSET $3
            ),
            episode_counts AS (
                SELECT
                    run_id,
                    COUNT(*)::INT as num_episodes
                FROM tensorzero.workflow_evaluation_run_episodes
                WHERE run_id IN (SELECT id FROM filtered_runs) AND staled_at IS NULL
                GROUP BY run_id
            )
            SELECT
                r.name,
                r.id,
                r.variant_pins,
                r.tags,
                r.project_name,
                COALESCE(e.num_episodes, 0)::INT as num_episodes,
                r.timestamp
            FROM filtered_runs r
            LEFT JOIN episode_counts e ON r.id = e.run_id
            ORDER BY r.id DESC
            ",
        );
    }

    #[test]
    fn test_build_list_workflow_evaluation_runs_query_with_project_name() {
        let qb = build_list_workflow_evaluation_runs_query(10, 0, None, Some("my_project"));
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            WITH filtered_runs AS (
                SELECT
                    run_display_name as name,
                    run_id as id,
                    variant_pins,
                    tags,
                    project_name,
                    created_at as timestamp
                FROM tensorzero.workflow_evaluation_runs
                WHERE staled_at IS NULL
                AND project_name = $1
                ORDER BY id DESC
                LIMIT $2 OFFSET $3
            ),
            episode_counts AS (
                SELECT
                    run_id,
                    COUNT(*)::INT as num_episodes
                FROM tensorzero.workflow_evaluation_run_episodes
                WHERE run_id IN (SELECT id FROM filtered_runs) AND staled_at IS NULL
                GROUP BY run_id
            )
            SELECT
                r.name,
                r.id,
                r.variant_pins,
                r.tags,
                r.project_name,
                COALESCE(e.num_episodes, 0)::INT as num_episodes,
                r.timestamp
            FROM filtered_runs r
            LEFT JOIN episode_counts e ON r.id = e.run_id
            ORDER BY r.id DESC
            ",
        );
    }

    #[test]
    fn test_build_get_workflow_evaluation_runs_query_no_project() {
        let run_ids = vec![Uuid::now_v7()];
        let qb = build_get_workflow_evaluation_runs_query(&run_ids, None);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_display_name as name,
                run_id as id,
                variant_pins,
                tags,
                project_name,
                created_at as timestamp
            FROM tensorzero.workflow_evaluation_runs
            WHERE staled_at IS NULL AND run_id = ANY($1)
            ORDER BY run_id DESC
            ",
        );
    }

    #[test]
    fn test_build_get_workflow_evaluation_runs_query_with_project() {
        let run_ids = vec![Uuid::now_v7()];
        let qb = build_get_workflow_evaluation_runs_query(&run_ids, Some("my_project"));
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                run_display_name as name,
                run_id as id,
                variant_pins,
                tags,
                project_name,
                created_at as timestamp
            FROM tensorzero.workflow_evaluation_runs
            WHERE staled_at IS NULL AND run_id = ANY($1)
            AND project_name = $2
            ORDER BY run_id DESC
            ",
        );
    }

    #[test]
    fn test_build_get_workflow_evaluation_run_statistics_query_no_filter() {
        let run_id = Uuid::now_v7();
        let qb = build_get_workflow_evaluation_run_statistics_query(run_id, None);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            WITH episodes AS (
                SELECT episode_id
                FROM tensorzero.workflow_evaluation_run_episodes
                WHERE run_id = $1 AND staled_at IS NULL
            ),
            float_stats AS (
                SELECT
                    metric_name,
                    COUNT(*)::INT as count,
                    AVG(value)::DOUBLE PRECISION as avg_metric,
                    STDDEV_SAMP(value)::DOUBLE PRECISION as stdev,
                    false as is_boolean
                FROM tensorzero.float_metric_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
                GROUP BY metric_name
            ),
            boolean_stats AS (
                SELECT
                    metric_name,
                    COUNT(*)::INT as count,
                    AVG(value::INT)::DOUBLE PRECISION as avg_metric,
                    STDDEV_SAMP(value::INT)::DOUBLE PRECISION as stdev,
                    true as is_boolean
                FROM tensorzero.boolean_metric_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
                GROUP BY metric_name
            )
            SELECT * FROM float_stats
            UNION ALL
            SELECT * FROM boolean_stats
            ORDER BY metric_name ASC
            ",
        );
    }

    #[test]
    fn test_build_get_workflow_evaluation_run_statistics_query_with_filter() {
        let run_id = Uuid::now_v7();
        let qb = build_get_workflow_evaluation_run_statistics_query(run_id, Some("solved"));
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            WITH episodes AS (
                SELECT episode_id
                FROM tensorzero.workflow_evaluation_run_episodes
                WHERE run_id = $1 AND staled_at IS NULL
            ),
            float_stats AS (
                SELECT
                    metric_name,
                    COUNT(*)::INT as count,
                    AVG(value)::DOUBLE PRECISION as avg_metric,
                    STDDEV_SAMP(value)::DOUBLE PRECISION as stdev,
                    false as is_boolean
                FROM tensorzero.float_metric_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
                AND metric_name = $2
                GROUP BY metric_name
            ),
            boolean_stats AS (
                SELECT
                    metric_name,
                    COUNT(*)::INT as count,
                    AVG(value::INT)::DOUBLE PRECISION as avg_metric,
                    STDDEV_SAMP(value::INT)::DOUBLE PRECISION as stdev,
                    true as is_boolean
                FROM tensorzero.boolean_metric_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
                AND metric_name = $3
                GROUP BY metric_name
            )
            SELECT * FROM float_stats
            UNION ALL
            SELECT * FROM boolean_stats
            ORDER BY metric_name ASC
            ",
        );
    }

    #[test]
    fn test_build_count_workflow_evaluation_run_episodes_by_task_name_query() {
        let run_ids = vec![Uuid::now_v7()];
        let qb = build_count_workflow_evaluation_run_episodes_by_task_name_query(&run_ids);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT COUNT(DISTINCT COALESCE(task_name, 'NULL_EPISODE_' || episode_id::TEXT))::INT as count
            FROM tensorzero.workflow_evaluation_run_episodes
            WHERE run_id = ANY($1) AND staled_at IS NULL
            ",
        );
    }

    #[test]
    fn test_build_count_workflow_evaluation_run_episodes_query() {
        let run_id = Uuid::now_v7();
        let qb = build_count_workflow_evaluation_run_episodes_query(run_id);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT COUNT(*)::INT as count
            FROM tensorzero.workflow_evaluation_run_episodes
            WHERE run_id = $1 AND staled_at IS NULL
            ",
        );
    }

    #[test]
    fn test_build_insert_workflow_evaluation_run_query() {
        let run_id = Uuid::now_v7();
        let variant_pins_json = serde_json::json!({});
        let tags_json = serde_json::json!({});
        let snapshot_hash_bytes = vec![0u8; 32];
        let created_at = Utc::now();

        let qb = build_insert_workflow_evaluation_run_query(
            run_id,
            &variant_pins_json,
            &tags_json,
            Some("my_project"),
            Some("my_run"),
            &snapshot_hash_bytes,
            created_at,
        );
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            INSERT INTO tensorzero.workflow_evaluation_runs
                (run_id, variant_pins, tags, project_name, run_display_name, snapshot_hash, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ",
        );
    }

    #[test]
    fn test_build_insert_workflow_evaluation_run_episode_query() {
        let episode_id = Uuid::now_v7();
        let run_id = Uuid::now_v7();
        let tags_json = serde_json::json!({});
        let snapshot_hash_bytes = vec![0u8; 32];
        let created_at = Utc::now();

        let qb = build_insert_workflow_evaluation_run_episode_query(
            episode_id,
            run_id,
            Some("my_task"),
            &tags_json,
            &snapshot_hash_bytes,
            created_at,
        );
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            INSERT INTO tensorzero.workflow_evaluation_run_episodes
                (episode_id, run_id, variant_pins, task_name, tags, snapshot_hash, created_at)
            SELECT
                $1, $2,
                r.variant_pins,
                $3, r.tags || $4, $5, $6
            FROM tensorzero.workflow_evaluation_runs r
            WHERE r.run_id = $7
            ",
        );
    }

    #[test]
    fn test_build_get_workflow_evaluation_run_by_episode_id_query() {
        let episode_id = Uuid::now_v7();
        let qb = build_get_workflow_evaluation_run_by_episode_id_query(episode_id);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT variant_pins, tags
            FROM tensorzero.workflow_evaluation_run_episodes
            WHERE episode_id = $1 AND staled_at IS NULL
            ",
        );
    }

    #[test]
    fn test_build_list_workflow_evaluation_run_episodes_by_task_name_query() {
        let run_ids = vec![Uuid::now_v7()];
        let qb = build_list_workflow_evaluation_run_episodes_by_task_name_query(&run_ids, 10, 0);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            WITH episodes_with_key AS (
                SELECT
                    episode_id,
                    run_id,
                    tags,
                    updated_at,
                    task_name,
                    COALESCE(task_name, 'NULL_EPISODE_' || episode_id::TEXT) as group_key
                FROM tensorzero.workflow_evaluation_run_episodes
                WHERE run_id = ANY($1) AND staled_at IS NULL
            ),
            group_keys AS (
                SELECT
                    group_key,
                    MAX(updated_at) as last_updated
                FROM episodes_with_key
                GROUP BY group_key
                ORDER BY last_updated DESC
                LIMIT $2 OFFSET $3
            ),
            episodes AS (
                SELECT e.*
                FROM episodes_with_key e
                WHERE e.group_key IN (SELECT group_key FROM group_keys)
            ),
            feedback_union AS (
                SELECT * FROM (
                    SELECT DISTINCT ON (target_id, metric_name)
                        target_id, metric_name, value::TEXT as value
                    FROM tensorzero.float_metric_feedback
                    WHERE target_id IN (SELECT episode_id FROM episodes)
                    ORDER BY target_id, metric_name, created_at DESC
                ) float_fb
                UNION ALL
                SELECT * FROM (
                    SELECT DISTINCT ON (target_id, metric_name)
                        target_id, metric_name, value::INT::TEXT as value
                    FROM tensorzero.boolean_metric_feedback
                    WHERE target_id IN (SELECT episode_id FROM episodes)
                    ORDER BY target_id, metric_name, created_at DESC
                ) bool_fb
                UNION ALL
                SELECT target_id, 'comment' as metric_name, value
                FROM tensorzero.comment_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
            )
            SELECT
                e.group_key,
                e.episode_id,
                e.updated_at as timestamp,
                e.run_id,
                e.tags,
                e.task_name,
                COALESCE(ARRAY_AGG(f.metric_name ORDER BY f.metric_name) FILTER (WHERE f.metric_name IS NOT NULL), ARRAY[]::TEXT[]) as feedback_metric_names,
                COALESCE(ARRAY_AGG(f.value ORDER BY f.metric_name) FILTER (WHERE f.metric_name IS NOT NULL), ARRAY[]::TEXT[]) as feedback_values
            FROM episodes e
            JOIN group_keys g ON e.group_key = g.group_key
            LEFT JOIN feedback_union f ON f.target_id = e.episode_id
            GROUP BY e.group_key, e.episode_id, e.run_id, e.tags, e.task_name, e.updated_at, g.last_updated
            ORDER BY g.last_updated DESC, e.group_key, e.episode_id
            ",
        );
    }

    #[test]
    fn test_build_get_workflow_evaluation_run_episodes_with_feedback_query() {
        let run_id = Uuid::now_v7();
        let qb = build_get_workflow_evaluation_run_episodes_with_feedback_query(run_id, 10, 0);
        let sql = qb.sql();
        let sql = sql.as_str();

        assert_query_equals(
            sql,
            r"
            WITH episodes AS (
                SELECT
                    episode_id,
                    run_id,
                    tags,
                    updated_at,
                    task_name
                FROM tensorzero.workflow_evaluation_run_episodes
                WHERE run_id = $1 AND staled_at IS NULL
                ORDER BY episode_id DESC
                LIMIT $2 OFFSET $3
            ),
            feedback_union AS (
                SELECT * FROM (
                    SELECT DISTINCT ON (target_id, metric_name)
                        target_id, metric_name, value::TEXT as value
                    FROM tensorzero.float_metric_feedback
                    WHERE target_id IN (SELECT episode_id FROM episodes)
                    ORDER BY target_id, metric_name, created_at DESC
                ) float_fb
                UNION ALL
                SELECT * FROM (
                    SELECT DISTINCT ON (target_id, metric_name)
                        target_id, metric_name, value::INT::TEXT as value
                    FROM tensorzero.boolean_metric_feedback
                    WHERE target_id IN (SELECT episode_id FROM episodes)
                    ORDER BY target_id, metric_name, created_at DESC
                ) bool_fb
                UNION ALL
                SELECT target_id, 'comment' as metric_name, value
                FROM tensorzero.comment_feedback
                WHERE target_id IN (SELECT episode_id FROM episodes)
            )
            SELECT
                e.episode_id,
                e.updated_at as timestamp,
                e.run_id,
                e.tags,
                e.task_name,
                COALESCE(ARRAY_AGG(f.metric_name ORDER BY f.metric_name) FILTER (WHERE f.metric_name IS NOT NULL), ARRAY[]::TEXT[]) as feedback_metric_names,
                COALESCE(ARRAY_AGG(f.value ORDER BY f.metric_name) FILTER (WHERE f.metric_name IS NOT NULL), ARRAY[]::TEXT[]) as feedback_values
            FROM episodes e
            LEFT JOIN feedback_union f ON f.target_id = e.episode_id
            GROUP BY e.episode_id, e.run_id, e.tags, e.task_name, e.updated_at
            ORDER BY e.episode_id DESC
            ",
        );
    }
}
