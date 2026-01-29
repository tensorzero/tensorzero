//! Feedback queries for Postgres.
//!
//! This module implements both read and write operations for feedback tables in Postgres.

use std::cmp::Reverse;
use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{PgPool, QueryBuilder, Row};
use uuid::Uuid;

use crate::config::{MetricConfigLevel, MetricConfigType};
use crate::db::feedback::{
    BooleanMetricFeedbackInsert, BooleanMetricFeedbackRow, CommentFeedbackInsert,
    CommentFeedbackRow, CommentTargetType, CumulativeFeedbackTimeSeriesPoint,
    DemonstrationFeedbackInsert, DemonstrationFeedbackRow, FeedbackBounds, FeedbackBoundsByType,
    FeedbackByVariant, FeedbackRow, FloatMetricFeedbackInsert, FloatMetricFeedbackRow,
    GetVariantPerformanceParams, InternalCumulativeFeedbackTimeSeriesPoint, LatestFeedbackRow,
    MetricType, MetricWithFeedback, StaticEvaluationHumanFeedbackInsert, VariantPerformanceRow,
};
use crate::db::{FeedbackQueries, TableBounds, TimeWindow};
use crate::error::{Error, ErrorDetails};
use crate::experimentation::asymptotic_confidence_sequences::asymp_cs;
use crate::function::FunctionConfig;

use super::PostgresConnectionInfo;

// =====================================================================
// FeedbackQueries trait implementation
// =====================================================================

#[async_trait]
impl FeedbackQueries for PostgresConnectionInfo {
    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: &str,
        variant_names: Option<&Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, Error> {
        let pool = self.get_pool_result()?;
        // Handle empty variant_names
        if let Some(names) = variant_names
            && names.is_empty()
        {
            return Ok(vec![]);
        }

        let mut qb = QueryBuilder::new(
            r"
    WITH feedback AS (
        SELECT target_id, value::INT::DOUBLE PRECISION as value
        FROM tensorzero.boolean_metric_feedback
        WHERE metric_name = ",
        );
        qb.push_bind(metric_name);
        qb.push(
            r"
        UNION ALL
        SELECT target_id, value
        FROM tensorzero.float_metric_feedback
        WHERE metric_name = ",
        );
        qb.push_bind(metric_name);
        qb.push(
            r"
    ),
    inferences AS (
        SELECT id, variant_name
        FROM tensorzero.chat_inferences
        WHERE function_name = ",
        );
        qb.push_bind(function_name);
        qb.push(
            r"
        UNION ALL
        SELECT id, variant_name
        FROM tensorzero.json_inferences
        WHERE function_name = ",
        );
        qb.push_bind(function_name);
        qb.push(
            r"
    )
    SELECT
        i.variant_name,
        AVG(f.value)::REAL as mean,
        VAR_SAMP(f.value)::REAL as variance,
        COUNT(*)::BIGINT as count
    FROM feedback f
    JOIN inferences i ON f.target_id = i.id
    WHERE 1=1",
        );

        if let Some(names) = variant_names {
            qb.push(" AND i.variant_name = ANY(");
            qb.push_bind(names);
            qb.push(")");
        }

        qb.push(" GROUP BY i.variant_name");

        let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

        let results = rows
            .into_iter()
            .map(|row| {
                let variance: Option<f32> = row.get("variance");
                FeedbackByVariant {
                    variant_name: row.get("variant_name"),
                    mean: row.get("mean"),
                    variance,
                    count: row.get::<i64, _>("count") as u64,
                }
            })
            .collect();

        Ok(results)
    }

    // TODO(#5927): Consider revisiting this query for performance.
    async fn get_cumulative_feedback_timeseries(
        &self,
        function_name: String,
        metric_name: String,
        variant_names: Option<Vec<String>>,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<CumulativeFeedbackTimeSeriesPoint>, Error> {
        let pool = self.get_pool_result()?;
        // Handle empty variant_names
        if let Some(ref names) = variant_names
            && names.is_empty()
        {
            return Ok(vec![]);
        }

        if time_window == TimeWindow::Cumulative {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cumulative time window is not supported for feedback timeseries"
                    .to_string(),
            }));
        }

        let interval_str = time_window.to_postgres_time_unit();

        // For this complex query with date_trunc expressions that need the interval as a literal,
        // we need to build separate queries for each time window type.
        // The interval string is trusted (from our enum match above).
        // Build the base CTE query using QueryBuilder for value bindings
        // Note: interval_str comes from a trusted enum match, so it's safe to include directly
        let mut qb = QueryBuilder::new("WITH feedback_with_variant AS (SELECT date_trunc('");
        qb.push(interval_str);
        qb.push("', f.created_at) + INTERVAL '1 ");
        qb.push(interval_str);
        qb.push(
            r"' AS period_end,
                i.variant_name,
                f.value
            FROM (
                SELECT target_id, value::INT::DOUBLE PRECISION as value, created_at
                FROM tensorzero.boolean_metric_feedback WHERE metric_name = ",
        );
        qb.push_bind(&metric_name);
        qb.push(
            r"
                UNION ALL
                SELECT target_id, value, created_at
                FROM tensorzero.float_metric_feedback WHERE metric_name = ",
        );
        qb.push_bind(&metric_name);
        qb.push(
            r"
            ) f
            JOIN (
                SELECT id, variant_name FROM tensorzero.chat_inferences WHERE function_name = ",
        );
        qb.push_bind(&function_name);
        qb.push(
            r"
                UNION ALL
                SELECT id, variant_name FROM tensorzero.json_inferences WHERE function_name = ",
        );
        qb.push_bind(&function_name);
        qb.push(") i ON f.target_id = i.id WHERE 1=1");

        if let Some(names) = variant_names {
            qb.push(" AND i.variant_name = ANY(");
            qb.push_bind(names.as_slice());
            qb.push(")");
        }

        qb.push(
        r"
        ),
        period_stats AS (
            SELECT
                period_end,
                variant_name,
                COUNT(*) as period_count,
                SUM(value) as period_sum,
                SUM(value * value) as period_sum_sq
            FROM feedback_with_variant
            GROUP BY period_end, variant_name
        ),
        cumulative_stats AS (
            SELECT
                period_end,
                variant_name,
                SUM(period_count) OVER w as count,
                SUM(period_sum) OVER w as sum_val,
                SUM(period_sum_sq) OVER w as sum_sq
            FROM period_stats
            WINDOW w AS (PARTITION BY variant_name ORDER BY period_end ROWS UNBOUNDED PRECEDING)
        ),
        max_period AS (
            SELECT MAX(period_end) as max_end FROM cumulative_stats
        )
        SELECT
            period_end,
            variant_name,
            (sum_val / count)::REAL as mean,
            CASE WHEN count > 1 THEN ((sum_sq - sum_val * sum_val / count) / (count - 1))::REAL END as variance,
            count::BIGINT
        FROM cumulative_stats
        WHERE period_end >= (SELECT max_end FROM max_period) - (",
    );
        qb.push_bind(max_periods as i32);
        qb.push(" || ' ");
        qb.push(interval_str);
        qb.push("')::INTERVAL ORDER BY period_end, variant_name");

        let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

        let internal_points: Vec<InternalCumulativeFeedbackTimeSeriesPoint> = rows
            .into_iter()
            .map(|row| {
                let period_end: DateTime<Utc> = row.get("period_end");
                let variance: Option<f32> = row.get("variance");
                InternalCumulativeFeedbackTimeSeriesPoint {
                    period_end,
                    variant_name: row.get("variant_name"),
                    mean: row.get("mean"),
                    variance,
                    count: row.get::<i64, _>("count") as u64,
                }
            })
            .collect();

        // Add confidence sequence bounds
        asymp_cs(internal_points, 0.05, None)
    }

    async fn query_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<Vec<FeedbackRow>, Error> {
        let pool = self.get_pool_result()?;
        if before.is_some() && after.is_some() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot specify both before and after in query_feedback_by_target_id"
                    .to_string(),
            }));
        }

        let limit = limit.unwrap_or(100) as i64;

        // Query all 4 feedback tables in parallel
        let (boolean_res, float_res, comment_res, demo_res) = tokio::join!(
            query_boolean_feedback(pool, target_id, before, after, limit),
            query_float_feedback(pool, target_id, before, after, limit),
            query_comment_feedback(pool, target_id, before, after, limit),
            query_demonstration_feedback(pool, target_id, before, after, limit),
        );

        let mut all_feedback: Vec<FeedbackRow> = Vec::new();
        all_feedback.extend(boolean_res?.into_iter().map(FeedbackRow::Boolean));
        all_feedback.extend(float_res?.into_iter().map(FeedbackRow::Float));
        all_feedback.extend(comment_res?.into_iter().map(FeedbackRow::Comment));
        all_feedback.extend(demo_res?.into_iter().map(FeedbackRow::Demonstration));

        // Sort by id descending (UUIDv7 sorts correctly)
        all_feedback.sort_by_key(|f| {
            Reverse(match f {
                FeedbackRow::Boolean(f) => f.id,
                FeedbackRow::Float(f) => f.id,
                FeedbackRow::Comment(f) => f.id,
                FeedbackRow::Demonstration(f) => f.id,
            })
        });

        // Truncate to limit
        all_feedback.truncate(limit as usize);

        Ok(all_feedback)
    }

    async fn query_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<FeedbackBounds, Error> {
        let pool = self.get_pool_result()?;
        // Query bounds for all 4 tables in parallel using static queries
        let (boolean, float, comment, demonstration) = tokio::join!(
            query_boolean_bounds(pool, target_id),
            query_float_bounds(pool, target_id),
            query_comment_bounds(pool, target_id),
            query_demonstration_bounds(pool, target_id),
        );

        let boolean = boolean?;
        let float = float?;
        let comment = comment?;
        let demonstration = demonstration?;

        // Compute overall first/last
        let all_ids: Vec<Uuid> = [
            boolean.first_id,
            boolean.last_id,
            float.first_id,
            float.last_id,
            comment.first_id,
            comment.last_id,
            demonstration.first_id,
            demonstration.last_id,
        ]
        .into_iter()
        .flatten()
        .collect();

        let first_id = all_ids.iter().min().copied();
        let last_id = all_ids.iter().max().copied();

        Ok(FeedbackBounds {
            first_id,
            last_id,
            by_type: FeedbackBoundsByType {
                boolean,
                float,
                comment,
                demonstration,
            },
        })
    }

    async fn count_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;
        let row = sqlx::query!(
            r#"
            SELECT (
                (SELECT COUNT(*) FROM tensorzero.boolean_metric_feedback WHERE target_id = $1) +
                (SELECT COUNT(*) FROM tensorzero.float_metric_feedback WHERE target_id = $1) +
                (SELECT COUNT(*) FROM tensorzero.comment_feedback WHERE target_id = $1) +
                (SELECT COUNT(*) FROM tensorzero.demonstration_feedback WHERE inference_id = $1)
            )::BIGINT as "total!"
            "#,
            target_id,
        )
        .fetch_one(pool)
        .await
        .map_err(Error::from)?;

        Ok(row.total as u64)
    }

    async fn query_demonstration_feedback_by_inference_id(
        &self,
        inference_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<Vec<DemonstrationFeedbackRow>, Error> {
        let pool = self.get_pool_result()?;
        let limit = limit.unwrap_or(100) as i64;
        query_demonstration_feedback(pool, inference_id, before, after, limit).await
    }

    async fn query_metrics_with_feedback(
        &self,
        function_name: &str,
        function_config: &FunctionConfig,
        variant_name: Option<&str>,
    ) -> Result<Vec<MetricWithFeedback>, Error> {
        let pool = self.get_pool_result()?;
        let table = function_config.postgres_table_name();

        // Build the query using QueryBuilder
        let mut qb = QueryBuilder::new("WITH inference_ids AS (SELECT id FROM ");
        qb.push(table);
        qb.push(" WHERE function_name = ");
        qb.push_bind(function_name);

        if let Some(vn) = variant_name {
            qb.push(" AND variant_name = ");
            qb.push_bind(vn);
        }

        qb.push(
            r")
            SELECT metric_name, feedback_count::INT FROM (
                SELECT metric_name, COUNT(*)::INT as feedback_count
                FROM tensorzero.boolean_metric_feedback
                WHERE target_id IN (SELECT id FROM inference_ids)
                GROUP BY metric_name
                UNION ALL
                SELECT metric_name, COUNT(*)::INT as feedback_count
                FROM tensorzero.float_metric_feedback
                WHERE target_id IN (SELECT id FROM inference_ids)
                GROUP BY metric_name
                UNION ALL
                SELECT 'demonstration' as metric_name, COUNT(*)::INT as feedback_count
                FROM tensorzero.demonstration_feedback
                WHERE inference_id IN (SELECT id FROM inference_ids)
            ) combined
            WHERE feedback_count > 0
            ORDER BY metric_name",
        );

        let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

        Ok(rows
            .into_iter()
            .map(|row| {
                let metric_name: String = row.get("metric_name");
                let metric_type = if metric_name == "demonstration" {
                    Some(MetricType::Demonstration)
                } else {
                    None
                };
                MetricWithFeedback {
                    function_name: function_name.to_string(),
                    metric_name,
                    metric_type,
                    feedback_count: row.get::<i32, _>("feedback_count") as u32,
                }
            })
            .collect())
    }

    async fn query_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<Vec<LatestFeedbackRow>, Error> {
        let pool = self.get_pool_result()?;
        let rows = sqlx::query!(
            r#"
            SELECT metric_name as "metric_name!", tensorzero.max_uuid(id)::TEXT as "latest_id!"
            FROM (
                SELECT metric_name, id FROM tensorzero.boolean_metric_feedback WHERE target_id = $1
                UNION ALL
                SELECT metric_name, id FROM tensorzero.float_metric_feedback WHERE target_id = $1
            ) combined
            GROUP BY metric_name
            ORDER BY metric_name
            "#,
            target_id,
        )
        .fetch_all(pool)
        .await
        .map_err(Error::from)?;

        Ok(rows
            .into_iter()
            .map(|row| LatestFeedbackRow {
                metric_name: row.metric_name,
                latest_id: row.latest_id,
            })
            .collect())
    }

    // TODO(#5927): Consider revisiting this query for performance.
    async fn get_variant_performances(
        &self,
        params: GetVariantPerformanceParams<'_>,
    ) -> Result<Vec<VariantPerformanceRow>, Error> {
        let pool = self.get_pool_result()?;
        let metric_level = params.metric_level();
        let inference_table = params.function_type.postgres_table_name();
        let metric_table = params.metric_config.r#type.postgres_table_name();

        // Boolean values need to be cast to INT first before DOUBLE PRECISION
        let value_cast = match params.metric_config.r#type {
            MetricConfigType::Boolean => "value::INT::DOUBLE PRECISION",
            MetricConfigType::Float => "value::DOUBLE PRECISION",
        };

        // Build time bucket expression based on time window
        let time_bucket_i = match params.time_window {
            TimeWindow::Minute => "date_trunc('minute', i.created_at)",
            TimeWindow::Hour => "date_trunc('hour', i.created_at)",
            TimeWindow::Day => "date_trunc('day', i.created_at)",
            TimeWindow::Week => "date_trunc('week', i.created_at)",
            TimeWindow::Month => "date_trunc('month', i.created_at)",
            TimeWindow::Cumulative => "'1970-01-01 00:00:00'::TIMESTAMPTZ",
        };

        // For both episode-level and inference-level metrics, we deduplicate feedback
        // by target_id to use only the latest feedback when multiple feedback entries
        // exist for the same target. This matches the ClickHouse implementation.
        //
        // For episode-level metrics, we additionally group by episode_id to avoid
        // counting the same episode multiple times when there are multiple inferences
        // per episode.
        let rows = match metric_level {
            MetricConfigLevel::Episode => {
                // Episode-level: deduplicate by grouping on episode_id in subquery
                let mut qb = QueryBuilder::new(
                    "WITH feedback AS (
                        SELECT DISTINCT ON (target_id) target_id, ",
                );
                qb.push(value_cast);
                qb.push(
                    " as value
                        FROM ",
                );
                qb.push(metric_table);
                qb.push(" WHERE metric_name = ");
                qb.push_bind(params.metric_name);
                qb.push(
                    " ORDER BY target_id, created_at DESC
                    ),
                    per_episode AS (
                        SELECT ",
                );
                qb.push(time_bucket_i);
                qb.push(
                    " AS period_start,
                            i.variant_name,
                            i.episode_id,
                            f.value
                        FROM ",
                );
                qb.push(inference_table);
                qb.push(
                    " i
                        JOIN feedback f ON f.target_id = i.episode_id
                        WHERE i.function_name = ",
                );
                qb.push_bind(params.function_name);

                if let Some(vn) = params.variant_name {
                    qb.push(" AND i.variant_name = ");
                    qb.push_bind(vn);
                }

                qb.push(
                    " GROUP BY period_start, i.variant_name, i.episode_id, f.value
                    )
                    SELECT
                        period_start,
                        variant_name,
                        COUNT(*)::INT as count,
                        AVG(value) as avg_metric,
                        STDDEV_SAMP(value) as stdev,
                        CASE WHEN COUNT(*) >= 2 THEN 1.96 * (STDDEV_SAMP(value) / SQRT(COUNT(*))) END as ci_error
                    FROM per_episode
                    GROUP BY period_start, variant_name
                    ORDER BY period_start ASC, variant_name ASC",
                );

                qb.build().fetch_all(pool).await.map_err(Error::from)?
            }
            MetricConfigLevel::Inference => {
                // Inference-level: SELECT DISTINCT ON (target_id) keeps the latest feedback by target_id.
                // (First result with SELECT DISTINCT ON target_id, ORDER BY created_at DESC)
                // This matches ClickHouse implementation.
                let mut qb = QueryBuilder::new(
                    "WITH feedback AS (
                        SELECT DISTINCT ON (target_id) target_id, ",
                );
                qb.push(value_cast);
                qb.push(
                    " as value, created_at
                        FROM ",
                );
                qb.push(metric_table);
                qb.push(" WHERE metric_name = ");
                qb.push_bind(params.metric_name);
                qb.push(
                    " ORDER BY target_id, created_at DESC
                    )
                    SELECT ",
                );
                qb.push(time_bucket_i);
                qb.push(
                    " as period_start,
                        i.variant_name,
                        COUNT(*)::INT as count,
                        AVG(f.value) as avg_metric,
                        STDDEV_SAMP(f.value) as stdev,
                        CASE WHEN COUNT(*) >= 2 THEN 1.96 * (STDDEV_SAMP(f.value) / SQRT(COUNT(*))) END as ci_error
                    FROM ",
                );
                qb.push(inference_table);
                qb.push(
                    " i
                    JOIN feedback f ON f.target_id = i.id
                    WHERE i.function_name = ",
                );
                qb.push_bind(params.function_name);

                if let Some(vn) = params.variant_name {
                    qb.push(" AND i.variant_name = ");
                    qb.push_bind(vn);
                }

                qb.push(
                    " GROUP BY period_start, i.variant_name ORDER BY period_start ASC, i.variant_name ASC",
                );

                qb.build().fetch_all(pool).await.map_err(Error::from)?
            }
        };

        Ok(rows
            .into_iter()
            .map(|row| VariantPerformanceRow {
                period_start: row.get("period_start"),
                variant_name: row.get("variant_name"),
                count: row.get::<i32, _>("count") as u32,
                avg_metric: row.get("avg_metric"),
                stdev: row.get("stdev"),
                ci_error: row.get("ci_error"),
            })
            .collect())
    }

    // Write methods
    async fn insert_boolean_feedback(
        &self,
        row: &BooleanMetricFeedbackInsert,
    ) -> Result<(), Error> {
        let pool = self.get_pool_result()?;
        let tags_json = serde_json::to_value(&row.tags).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize tags: {e}"),
            })
        })?;
        sqlx::query!(
            r#"
            INSERT INTO tensorzero.boolean_metric_feedback (id, target_id, metric_name, value, tags, snapshot_hash)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
            row.id,
            row.target_id,
            row.metric_name,
            row.value,
            tags_json,
            row.snapshot_hash.as_bytes(),
        )
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresConnection {
                message: format!("Failed to insert boolean feedback: {e}"),
            })
        })?;

        Ok(())
    }

    async fn insert_float_feedback(&self, row: &FloatMetricFeedbackInsert) -> Result<(), Error> {
        let pool = self.get_pool_result()?;
        let tags_json = serde_json::to_value(&row.tags).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize tags: {e}"),
            })
        })?;
        sqlx::query!(
            r#"
            INSERT INTO tensorzero.float_metric_feedback (id, target_id, metric_name, value, tags, snapshot_hash)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
            row.id,
            row.target_id,
            row.metric_name,
            row.value,
            tags_json,
            row.snapshot_hash.as_bytes(),
        )
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresConnection {
                message: format!("Failed to insert float feedback: {e}"),
            })
        })?;

        Ok(())
    }

    async fn insert_comment_feedback(&self, row: &CommentFeedbackInsert) -> Result<(), Error> {
        let pool = self.get_pool_result()?;
        let tags_json = serde_json::to_value(&row.tags).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize tags: {e}"),
            })
        })?;
        let target_type_str = match row.target_type {
            CommentTargetType::Inference => "inference",
            CommentTargetType::Episode => "episode",
        };

        sqlx::query!(
            r#"
            INSERT INTO tensorzero.comment_feedback (id, target_id, target_type, value, tags, snapshot_hash)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
            row.id,
            row.target_id,
            target_type_str,
            row.value,
            tags_json,
            row.snapshot_hash.as_bytes(),
        )
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresConnection {
                message: format!("Failed to insert comment feedback: {e}"),
            })
        })?;

        Ok(())
    }

    async fn insert_demonstration_feedback(
        &self,
        row: &DemonstrationFeedbackInsert,
    ) -> Result<(), Error> {
        let pool = self.get_pool_result()?;
        let tags_json = serde_json::to_value(&row.tags).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize tags: {e}"),
            })
        })?;
        // Parse the value string as JSON for JSONB storage
        let value_json: serde_json::Value = serde_json::from_str(&row.value).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to parse demonstration value as JSON: {e}"),
            })
        })?;
        sqlx::query!(
            r#"
            INSERT INTO tensorzero.demonstration_feedback (id, inference_id, value, tags, snapshot_hash)
            VALUES ($1, $2, $3, $4, $5)
            "#,
            row.id,
            row.inference_id,
            value_json,
            tags_json,
            row.snapshot_hash.as_bytes(),
        )
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresConnection {
                message: format!("Failed to insert demonstration feedback: {e}"),
            })
        })?;

        Ok(())
    }

    async fn insert_static_eval_feedback(
        &self,
        row: &StaticEvaluationHumanFeedbackInsert,
    ) -> Result<(), Error> {
        let pool = self.get_pool_result()?;
        sqlx::query!(
            r#"
            INSERT INTO tensorzero.inference_evaluation_human_feedback
                (feedback_id, metric_name, datapoint_id, output, value, evaluator_inference_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
            row.feedback_id,
            row.metric_name,
            row.datapoint_id,
            row.output,
            row.value,
            row.evaluator_inference_id,
        )
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresConnection {
                message: format!("Failed to insert static evaluation human feedback: {e}"),
            })
        })?;

        Ok(())
    }
}

async fn query_boolean_feedback(
    pool: &PgPool,
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<Vec<BooleanMetricFeedbackRow>, Error> {
    let mut qb = QueryBuilder::new(
        "SELECT id, target_id, metric_name, value, tags, created_at FROM tensorzero.boolean_metric_feedback WHERE target_id = ",
    );
    qb.push_bind(target_id);

    add_pagination_clause(&mut qb, before, after)?;

    qb.push(" LIMIT ");
    qb.push_bind(limit);

    let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

    rows.into_iter()
        .map(|row| {
            let tags_json: serde_json::Value = row.get("tags");
            let tags: HashMap<String, String> =
                serde_json::from_value(tags_json).unwrap_or_default();
            Ok(BooleanMetricFeedbackRow {
                id: row.get("id"),
                target_id: row.get("target_id"),
                metric_name: row.get("metric_name"),
                value: row.get("value"),
                tags,
                timestamp: row.get("created_at"),
            })
        })
        .collect()
}

async fn query_float_feedback(
    pool: &PgPool,
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<Vec<FloatMetricFeedbackRow>, Error> {
    let mut qb = QueryBuilder::new(
        "SELECT id, target_id, metric_name, value, tags, created_at FROM tensorzero.float_metric_feedback WHERE target_id = ",
    );
    qb.push_bind(target_id);

    add_pagination_clause(&mut qb, before, after)?;

    qb.push(" LIMIT ");
    qb.push_bind(limit);

    let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

    rows.into_iter()
        .map(|row| {
            let tags_json: serde_json::Value = row.get("tags");
            let tags: HashMap<String, String> =
                serde_json::from_value(tags_json).unwrap_or_default();
            Ok(FloatMetricFeedbackRow {
                id: row.get("id"),
                target_id: row.get("target_id"),
                metric_name: row.get("metric_name"),
                value: row.get("value"),
                tags,
                timestamp: row.get("created_at"),
            })
        })
        .collect()
}

async fn query_comment_feedback(
    pool: &PgPool,
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<Vec<CommentFeedbackRow>, Error> {
    let mut qb = QueryBuilder::new(
        "SELECT id, target_id, target_type, value, tags, created_at FROM tensorzero.comment_feedback WHERE target_id = ",
    );
    qb.push_bind(target_id);

    add_pagination_clause(&mut qb, before, after)?;

    qb.push(" LIMIT ");
    qb.push_bind(limit);

    let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

    rows.into_iter()
        .map(|row| {
            let tags_json: serde_json::Value = row.get("tags");
            let tags: HashMap<String, String> =
                serde_json::from_value(tags_json).unwrap_or_default();
            let target_type_str: String = row.get("target_type");
            let target_type = match target_type_str.as_str() {
                "inference" => CommentTargetType::Inference,
                "episode" => CommentTargetType::Episode,
                _ => CommentTargetType::Inference,
            };
            Ok(CommentFeedbackRow {
                id: row.get("id"),
                target_id: row.get("target_id"),
                target_type,
                value: row.get("value"),
                tags,
                timestamp: row.get("created_at"),
            })
        })
        .collect()
}

async fn query_demonstration_feedback(
    pool: &PgPool,
    inference_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<Vec<DemonstrationFeedbackRow>, Error> {
    let mut qb = QueryBuilder::new(
        "SELECT id, inference_id, value, tags, created_at FROM tensorzero.demonstration_feedback WHERE inference_id = ",
    );
    qb.push_bind(inference_id);

    add_pagination_clause(&mut qb, before, after)?;

    qb.push(" LIMIT ");
    qb.push_bind(limit);

    let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

    rows.into_iter()
        .map(|row| {
            let tags_json: serde_json::Value = row.get("tags");
            let tags: HashMap<String, String> =
                serde_json::from_value(tags_json).unwrap_or_default();
            // value is stored as JSONB, convert back to String
            // TODO(#5691): Remove this conversion after we align the types between ClickHouse and Postgres.
            let value_json: serde_json::Value = row.get("value");
            let value = serde_json::to_string(&value_json).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize demonstration value: {e}"),
                })
            })?;
            Ok(DemonstrationFeedbackRow {
                id: row.get("id"),
                inference_id: row.get("inference_id"),
                value,
                tags,
                timestamp: row.get("created_at"),
            })
        })
        .collect()
}

/// Adds pagination clause to query builder and returns the sort order
fn add_pagination_clause(
    qb: &mut QueryBuilder<sqlx::Postgres>,
    before: Option<Uuid>,
    after: Option<Uuid>,
) -> Result<(), Error> {
    match (before, after) {
        (Some(before), None) => {
            qb.push(" AND id < ");
            qb.push_bind(before);
            qb.push(" ORDER BY id DESC");
        }
        (None, Some(after)) => {
            qb.push(" AND id > ");
            qb.push_bind(after);
            qb.push(" ORDER BY id ASC");
        }
        (Some(_before), Some(_after)) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot specify both before and after in add_pagination_clause"
                    .to_string(),
            }));
        }
        (None, None) => {
            qb.push(" ORDER BY id DESC");
        }
    }
    Ok(())
}

async fn query_boolean_bounds(pool: &PgPool, target_id: Uuid) -> Result<TableBounds, Error> {
    let row = sqlx::query!(
        r#"
        SELECT
            tensorzero.min_uuid(id) as first_id,
            tensorzero.max_uuid(id) as last_id
        FROM tensorzero.boolean_metric_feedback
        WHERE target_id = $1
        "#,
        target_id,
    )
    .fetch_one(pool)
    .await
    .map_err(Error::from)?;

    Ok(TableBounds {
        first_id: row.first_id,
        last_id: row.last_id,
    })
}

async fn query_float_bounds(pool: &PgPool, target_id: Uuid) -> Result<TableBounds, Error> {
    let row = sqlx::query!(
        r#"
        SELECT
            tensorzero.min_uuid(id) as first_id,
            tensorzero.max_uuid(id) as last_id
        FROM tensorzero.float_metric_feedback
        WHERE target_id = $1
        "#,
        target_id,
    )
    .fetch_one(pool)
    .await
    .map_err(Error::from)?;

    Ok(TableBounds {
        first_id: row.first_id,
        last_id: row.last_id,
    })
}

async fn query_comment_bounds(pool: &PgPool, target_id: Uuid) -> Result<TableBounds, Error> {
    let row = sqlx::query!(
        r#"
        SELECT
            tensorzero.min_uuid(id) as first_id,
            tensorzero.max_uuid(id) as last_id
        FROM tensorzero.comment_feedback
        WHERE target_id = $1
        "#,
        target_id,
    )
    .fetch_one(pool)
    .await
    .map_err(Error::from)?;

    Ok(TableBounds {
        first_id: row.first_id,
        last_id: row.last_id,
    })
}

async fn query_demonstration_bounds(pool: &PgPool, target_id: Uuid) -> Result<TableBounds, Error> {
    let row = sqlx::query!(
        r#"
        SELECT
            tensorzero.min_uuid(id) as first_id,
            tensorzero.max_uuid(id) as last_id
        FROM tensorzero.demonstration_feedback
        WHERE inference_id = $1
        "#,
        target_id,
    )
    .fetch_one(pool)
    .await
    .map_err(Error::from)?;

    Ok(TableBounds {
        first_id: row.first_id,
        last_id: row.last_id,
    })
}
