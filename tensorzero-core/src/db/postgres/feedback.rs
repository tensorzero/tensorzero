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
// Query builder functions (for unit testing)
// =====================================================================

/// Builds a query to get feedback statistics grouped by variant.
///
/// If `variant_names` is empty, no variant filter is applied (all variants are included).
/// If `variant_names` is non-empty, only the specified variants are included.
///
/// If `namespace` is `Some`, the inferences CTE is extended to include both inference-level
/// and episode-level IDs filtered by `tags->>'tensorzero::namespace'`.
///
/// If `max_samples_per_variant` is `Some`, uses `ROW_NUMBER() OVER (PARTITION BY variant_name)`
/// to limit to the most recent N samples per variant before aggregating.
fn build_feedback_by_variant_query(
    metric_name: &str,
    function_name: &str,
    variant_names: &[String],
    namespace: Option<&str>,
    max_samples_per_variant: Option<u64>,
) -> QueryBuilder<sqlx::Postgres> {
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

    if let Some(ns) = namespace {
        // Namespace-filtered: include inference IDs and episode IDs from inferences
        // that match the namespace tag
        qb.push(
            r"
    ),
    inferences AS (
        SELECT id, variant_name
        FROM tensorzero.chat_inferences
        WHERE function_name = ",
        );
        qb.push_bind(function_name);
        qb.push(r" AND tags->>'tensorzero::namespace' = ");
        qb.push_bind(ns);
        qb.push(
            r"
        UNION ALL
        SELECT id, variant_name
        FROM tensorzero.json_inferences
        WHERE function_name = ",
        );
        qb.push_bind(function_name);
        qb.push(r" AND tags->>'tensorzero::namespace' = ");
        qb.push_bind(ns);
        qb.push(
            r"
        UNION ALL
        SELECT DISTINCT episode_id AS id, variant_name
        FROM tensorzero.chat_inferences
        WHERE function_name = ",
        );
        qb.push_bind(function_name);
        qb.push(r" AND tags->>'tensorzero::namespace' = ");
        qb.push_bind(ns);
        qb.push(
            r" AND episode_id IS NOT NULL
        UNION ALL
        SELECT DISTINCT episode_id AS id, variant_name
        FROM tensorzero.json_inferences
        WHERE function_name = ",
        );
        qb.push_bind(function_name);
        qb.push(r" AND tags->>'tensorzero::namespace' = ");
        qb.push_bind(ns);
        qb.push(" AND episode_id IS NOT NULL");
    } else {
        // No namespace filter: standard inference-level join
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
    }

    qb.push(
        r"
    )",
    );

    if let Some(limit) = max_samples_per_variant {
        // Use ROW_NUMBER() to limit samples per variant before aggregating
        qb.push(
            r",
    numbered AS (
        SELECT
            i.variant_name,
            f.value,
            ROW_NUMBER() OVER (PARTITION BY i.variant_name ORDER BY f.target_id DESC) as rn
        FROM feedback f
        JOIN inferences i ON f.target_id = i.id
        WHERE 1=1",
        );

        if !variant_names.is_empty() {
            qb.push(" AND i.variant_name = ANY(");
            qb.push_bind(variant_names);
            qb.push(")");
        }

        qb.push(
            r"
    )
    SELECT
        variant_name,
        AVG(value)::REAL as mean,
        VAR_SAMP(value)::REAL as variance,
        COUNT(*)::BIGINT as count
    FROM numbered
    WHERE rn <= ",
        );
        qb.push_bind(limit as i64);
        qb.push(" GROUP BY variant_name");
    } else {
        // No sample limit: aggregate directly
        qb.push(
            r"
    SELECT
        i.variant_name,
        AVG(f.value)::REAL as mean,
        VAR_SAMP(f.value)::REAL as variance,
        COUNT(*)::BIGINT as count
    FROM feedback f
    JOIN inferences i ON f.target_id = i.id
    WHERE 1=1",
        );

        if !variant_names.is_empty() {
            qb.push(" AND i.variant_name = ANY(");
            qb.push_bind(variant_names);
            qb.push(")");
        }

        qb.push(" GROUP BY i.variant_name");
    }

    qb
}

/// Builds a query for cumulative feedback time series.
///
/// The `interval_str` parameter must be a trusted value from `TimeWindow::to_postgres_time_unit()`.
///
/// If `variant_names` is empty, no variant filter is applied (all variants are included).
/// If `variant_names` is non-empty, only the specified variants are included.
fn build_cumulative_feedback_timeseries_query(
    metric_name: &str,
    function_name: &str,
    variant_names: &[String],
    interval_str: &str,
    max_periods: i32,
) -> QueryBuilder<sqlx::Postgres> {
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
    qb.push_bind(metric_name);
    qb.push(
        r"
            UNION ALL
            SELECT target_id, value, created_at
            FROM tensorzero.float_metric_feedback WHERE metric_name = ",
    );
    qb.push_bind(metric_name);
    qb.push(
        r"
        ) f
        JOIN (
            SELECT id, variant_name FROM tensorzero.chat_inferences WHERE function_name = ",
    );
    qb.push_bind(function_name);
    qb.push(
        r"
            UNION ALL
            SELECT id, variant_name FROM tensorzero.json_inferences WHERE function_name = ",
    );
    qb.push_bind(function_name);
    qb.push(") i ON f.target_id = i.id WHERE 1=1");

    if !variant_names.is_empty() {
        qb.push(" AND i.variant_name = ANY(");
        qb.push_bind(variant_names);
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
    qb.push_bind(max_periods);
    qb.push(" || ' ");
    qb.push(interval_str);
    qb.push("')::INTERVAL ORDER BY period_end, variant_name");

    qb
}

/// Builds a query to get metrics that have feedback for a function.
fn build_metrics_with_feedback_query(
    function_name: &str,
    table: &str,
    variant_name: Option<&str>,
) -> QueryBuilder<sqlx::Postgres> {
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

    qb
}

/// Parameters for building a variant performances query.
struct VariantPerformancesQueryParams<'a> {
    metric_name: &'a str,
    function_name: &'a str,
    inference_table: &'a str,
    metric_table: &'a str,
    value_cast: &'a str,
    time_bucket_expr: &'a str,
    metric_level: MetricConfigLevel,
    variant_name: Option<&'a str>,
}

/// Builds a query for variant performances.
fn build_variant_performances_query(
    params: &VariantPerformancesQueryParams<'_>,
) -> QueryBuilder<sqlx::Postgres> {
    match params.metric_level {
        MetricConfigLevel::Episode => build_variant_performances_episode_query(params),
        MetricConfigLevel::Inference => build_variant_performances_inference_query(params),
    }
}

fn build_variant_performances_episode_query(
    params: &VariantPerformancesQueryParams<'_>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        "WITH feedback AS (
            SELECT DISTINCT ON (target_id) target_id, ",
    );
    qb.push(params.value_cast);
    qb.push(
        " as value
            FROM ",
    );
    qb.push(params.metric_table);
    qb.push(" WHERE metric_name = ");
    qb.push_bind(params.metric_name);
    qb.push(
        " ORDER BY target_id, created_at DESC
        ),
        per_episode AS (
            SELECT ",
    );
    qb.push(params.time_bucket_expr);
    qb.push(
        " AS period_start,
                i.variant_name,
                i.episode_id,
                f.value
            FROM ",
    );
    qb.push(params.inference_table);
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

    qb
}

fn build_variant_performances_inference_query(
    params: &VariantPerformancesQueryParams<'_>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        "WITH feedback AS (
            SELECT DISTINCT ON (target_id) target_id, ",
    );
    qb.push(params.value_cast);
    qb.push(
        " as value, created_at
            FROM ",
    );
    qb.push(params.metric_table);
    qb.push(" WHERE metric_name = ");
    qb.push_bind(params.metric_name);
    qb.push(
        " ORDER BY target_id, created_at DESC
        )
        SELECT ",
    );
    qb.push(params.time_bucket_expr);
    qb.push(
        " as period_start,
            i.variant_name,
            COUNT(*)::INT as count,
            AVG(f.value) as avg_metric,
            STDDEV_SAMP(f.value) as stdev,
            CASE WHEN COUNT(*) >= 2 THEN 1.96 * (STDDEV_SAMP(f.value) / SQRT(COUNT(*))) END as ci_error
        FROM ",
    );
    qb.push(params.inference_table);
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

    qb.push(" GROUP BY period_start, i.variant_name ORDER BY period_start ASC, i.variant_name ASC");

    qb
}

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
        namespace: Option<&str>,
        max_samples_per_variant: Option<u64>,
    ) -> Result<Vec<FeedbackByVariant>, Error> {
        let pool = self.get_pool_result()?;
        // Handle empty variant_names - return early to avoid unnecessary query
        if let Some(names) = variant_names
            && names.is_empty()
        {
            return Ok(vec![]);
        }

        // Pass empty slice for None, or the actual slice for Some
        let variant_slice = variant_names.map(|v| v.as_slice()).unwrap_or(&[]);
        let mut qb = build_feedback_by_variant_query(
            metric_name,
            function_name,
            variant_slice,
            namespace,
            max_samples_per_variant,
        );

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

        // Pass empty slice for None, or the actual slice for Some
        let variant_slice = variant_names.as_deref().unwrap_or(&[]);
        let mut qb = build_cumulative_feedback_timeseries_query(
            &metric_name,
            &function_name,
            variant_slice,
            interval_str,
            max_periods as i32,
        );

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

        let mut qb = build_metrics_with_feedback_query(function_name, table, variant_name);

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
        let time_bucket_expr = match params.time_window {
            TimeWindow::Minute => "date_trunc('minute', i.created_at)",
            TimeWindow::Hour => "date_trunc('hour', i.created_at)",
            TimeWindow::Day => "date_trunc('day', i.created_at)",
            TimeWindow::Week => "date_trunc('week', i.created_at)",
            TimeWindow::Month => "date_trunc('month', i.created_at)",
            TimeWindow::Cumulative => "'1970-01-01 00:00:00'::TIMESTAMPTZ",
        };

        let query_params = VariantPerformancesQueryParams {
            metric_name: params.metric_name,
            function_name: params.function_name,
            inference_table,
            metric_table,
            value_cast,
            time_bucket_expr,
            metric_level,
            variant_name: params.variant_name,
        };

        let mut qb = build_variant_performances_query(&query_params);
        let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

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

/// Builds a query to fetch boolean metric feedback for a target.
fn build_boolean_feedback_query(
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<QueryBuilder<sqlx::Postgres>, Error> {
    let mut qb = QueryBuilder::new(
        "SELECT id, target_id, metric_name, value, tags, created_at FROM tensorzero.boolean_metric_feedback WHERE target_id = ",
    );
    qb.push_bind(target_id);

    add_pagination_clause(&mut qb, before, after)?;

    qb.push(" LIMIT ");
    qb.push_bind(limit);

    Ok(qb)
}

async fn query_boolean_feedback(
    pool: &PgPool,
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<Vec<BooleanMetricFeedbackRow>, Error> {
    let mut qb = build_boolean_feedback_query(target_id, before, after, limit)?;

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

/// Builds a query to fetch float metric feedback for a target.
fn build_float_feedback_query(
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<QueryBuilder<sqlx::Postgres>, Error> {
    let mut qb = QueryBuilder::new(
        "SELECT id, target_id, metric_name, value, tags, created_at FROM tensorzero.float_metric_feedback WHERE target_id = ",
    );
    qb.push_bind(target_id);

    add_pagination_clause(&mut qb, before, after)?;

    qb.push(" LIMIT ");
    qb.push_bind(limit);

    Ok(qb)
}

async fn query_float_feedback(
    pool: &PgPool,
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<Vec<FloatMetricFeedbackRow>, Error> {
    let mut qb = build_float_feedback_query(target_id, before, after, limit)?;

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

/// Builds a query to fetch comment feedback for a target.
fn build_comment_feedback_query(
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<QueryBuilder<sqlx::Postgres>, Error> {
    let mut qb = QueryBuilder::new(
        "SELECT id, target_id, target_type, value, tags, created_at FROM tensorzero.comment_feedback WHERE target_id = ",
    );
    qb.push_bind(target_id);

    add_pagination_clause(&mut qb, before, after)?;

    qb.push(" LIMIT ");
    qb.push_bind(limit);

    Ok(qb)
}

async fn query_comment_feedback(
    pool: &PgPool,
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<Vec<CommentFeedbackRow>, Error> {
    let mut qb = build_comment_feedback_query(target_id, before, after, limit)?;

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

/// Builds a query to fetch demonstration feedback for an inference.
fn build_demonstration_feedback_query(
    inference_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<QueryBuilder<sqlx::Postgres>, Error> {
    let mut qb = QueryBuilder::new(
        "SELECT id, inference_id, value, tags, created_at FROM tensorzero.demonstration_feedback WHERE inference_id = ",
    );
    qb.push_bind(inference_id);

    add_pagination_clause(&mut qb, before, after)?;

    qb.push(" LIMIT ");
    qb.push_bind(limit);

    Ok(qb)
}

async fn query_demonstration_feedback(
    pool: &PgPool,
    inference_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: i64,
) -> Result<Vec<DemonstrationFeedbackRow>, Error> {
    let mut qb = build_demonstration_feedback_query(inference_id, before, after, limit)?;

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

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_helpers::assert_query_equals;

    // ===== build_boolean_feedback_query tests =====

    #[test]
    fn test_build_boolean_feedback_query_no_pagination() {
        let target_id = Uuid::now_v7();
        let qb = build_boolean_feedback_query(target_id, None, None, 100).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, target_id, metric_name, value, tags, created_at
            FROM tensorzero.boolean_metric_feedback
            WHERE target_id = $1 ORDER BY id DESC LIMIT $2
            ",
        );
    }

    #[test]
    fn test_build_boolean_feedback_query_before_cursor() {
        let target_id = Uuid::now_v7();
        let before_id = Uuid::now_v7();
        let qb = build_boolean_feedback_query(target_id, Some(before_id), None, 50).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, target_id, metric_name, value, tags, created_at
            FROM tensorzero.boolean_metric_feedback
            WHERE target_id = $1 AND id < $2 ORDER BY id DESC LIMIT $3
            ",
        );
    }

    #[test]
    fn test_build_boolean_feedback_query_after_cursor() {
        let target_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();
        let qb = build_boolean_feedback_query(target_id, None, Some(after_id), 50).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, target_id, metric_name, value, tags, created_at
            FROM tensorzero.boolean_metric_feedback
            WHERE target_id = $1 AND id > $2 ORDER BY id ASC LIMIT $3
            ",
        );
    }

    // ===== build_float_feedback_query tests =====

    #[test]
    fn test_build_float_feedback_query_no_pagination() {
        let target_id = Uuid::now_v7();
        let qb = build_float_feedback_query(target_id, None, None, 100).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, target_id, metric_name, value, tags, created_at
            FROM tensorzero.float_metric_feedback
            WHERE target_id = $1 ORDER BY id DESC LIMIT $2
            ",
        );
    }

    #[test]
    fn test_build_float_feedback_query_before_cursor() {
        let target_id = Uuid::now_v7();
        let before_id = Uuid::now_v7();
        let qb = build_float_feedback_query(target_id, Some(before_id), None, 50).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, target_id, metric_name, value, tags, created_at
            FROM tensorzero.float_metric_feedback
            WHERE target_id = $1 AND id < $2 ORDER BY id DESC LIMIT $3
            ",
        );
    }

    #[test]
    fn test_build_float_feedback_query_after_cursor() {
        let target_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();
        let qb = build_float_feedback_query(target_id, None, Some(after_id), 50).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, target_id, metric_name, value, tags, created_at
            FROM tensorzero.float_metric_feedback
            WHERE target_id = $1 AND id > $2 ORDER BY id ASC LIMIT $3
            ",
        );
    }

    // ===== build_comment_feedback_query tests =====

    #[test]
    fn test_build_comment_feedback_query_no_pagination() {
        let target_id = Uuid::now_v7();
        let qb = build_comment_feedback_query(target_id, None, None, 100).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, target_id, target_type, value, tags, created_at
            FROM tensorzero.comment_feedback
            WHERE target_id = $1 ORDER BY id DESC LIMIT $2
            ",
        );
    }

    #[test]
    fn test_build_comment_feedback_query_before_cursor() {
        let target_id = Uuid::now_v7();
        let before_id = Uuid::now_v7();
        let qb = build_comment_feedback_query(target_id, Some(before_id), None, 50).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, target_id, target_type, value, tags, created_at
            FROM tensorzero.comment_feedback
            WHERE target_id = $1 AND id < $2 ORDER BY id DESC LIMIT $3
            ",
        );
    }

    #[test]
    fn test_build_comment_feedback_query_after_cursor() {
        let target_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();
        let qb = build_comment_feedback_query(target_id, None, Some(after_id), 50).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, target_id, target_type, value, tags, created_at
            FROM tensorzero.comment_feedback
            WHERE target_id = $1 AND id > $2 ORDER BY id ASC LIMIT $3
            ",
        );
    }

    // ===== build_demonstration_feedback_query tests =====

    #[test]
    fn test_build_demonstration_feedback_query_no_pagination() {
        let inference_id = Uuid::now_v7();
        let qb = build_demonstration_feedback_query(inference_id, None, None, 100).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, inference_id, value, tags, created_at
            FROM tensorzero.demonstration_feedback
            WHERE inference_id = $1 ORDER BY id DESC LIMIT $2
            ",
        );
    }

    #[test]
    fn test_build_demonstration_feedback_query_before_cursor() {
        let inference_id = Uuid::now_v7();
        let before_id = Uuid::now_v7();
        let qb =
            build_demonstration_feedback_query(inference_id, Some(before_id), None, 50).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, inference_id, value, tags, created_at
            FROM tensorzero.demonstration_feedback
            WHERE inference_id = $1 AND id < $2 ORDER BY id DESC LIMIT $3
            ",
        );
    }

    #[test]
    fn test_build_demonstration_feedback_query_after_cursor() {
        let inference_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();
        let qb =
            build_demonstration_feedback_query(inference_id, None, Some(after_id), 50).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id, inference_id, value, tags, created_at
            FROM tensorzero.demonstration_feedback
            WHERE inference_id = $1 AND id > $2 ORDER BY id ASC LIMIT $3
            ",
        );
    }

    // ===== build_feedback_by_variant_query tests =====

    #[test]
    fn test_build_feedback_by_variant_query_no_variant_filter() {
        let qb = build_feedback_by_variant_query("my_metric", "my_function", &[], None, None);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT target_id, value::INT::DOUBLE PRECISION as value
                FROM tensorzero.boolean_metric_feedback
                WHERE metric_name = $1
                UNION ALL
                SELECT target_id, value
                FROM tensorzero.float_metric_feedback
                WHERE metric_name = $2
            ),
            inferences AS (
                SELECT id, variant_name
                FROM tensorzero.chat_inferences
                WHERE function_name = $3
                UNION ALL
                SELECT id, variant_name
                FROM tensorzero.json_inferences
                WHERE function_name = $4
            )
            SELECT
                i.variant_name,
                AVG(f.value)::REAL as mean,
                VAR_SAMP(f.value)::REAL as variance,
                COUNT(*)::BIGINT as count
            FROM feedback f
            JOIN inferences i ON f.target_id = i.id
            WHERE 1=1 GROUP BY i.variant_name
            ",
        );
    }

    #[test]
    fn test_build_feedback_by_variant_query_with_variant_filter() {
        let variant_names = vec!["variant_a".to_string(), "variant_b".to_string()];
        let qb =
            build_feedback_by_variant_query("my_metric", "my_function", &variant_names, None, None);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT target_id, value::INT::DOUBLE PRECISION as value
                FROM tensorzero.boolean_metric_feedback
                WHERE metric_name = $1
                UNION ALL
                SELECT target_id, value
                FROM tensorzero.float_metric_feedback
                WHERE metric_name = $2
            ),
            inferences AS (
                SELECT id, variant_name
                FROM tensorzero.chat_inferences
                WHERE function_name = $3
                UNION ALL
                SELECT id, variant_name
                FROM tensorzero.json_inferences
                WHERE function_name = $4
            )
            SELECT
                i.variant_name,
                AVG(f.value)::REAL as mean,
                VAR_SAMP(f.value)::REAL as variance,
                COUNT(*)::BIGINT as count
            FROM feedback f
            JOIN inferences i ON f.target_id = i.id
            WHERE 1=1 AND i.variant_name = ANY($5) GROUP BY i.variant_name
            ",
        );
    }

    #[test]
    fn test_build_feedback_by_variant_query_with_namespace() {
        let qb = build_feedback_by_variant_query(
            "my_metric",
            "my_function",
            &[],
            Some("production"),
            None,
        );
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT target_id, value::INT::DOUBLE PRECISION as value
                FROM tensorzero.boolean_metric_feedback
                WHERE metric_name = $1
                UNION ALL
                SELECT target_id, value
                FROM tensorzero.float_metric_feedback
                WHERE metric_name = $2
            ),
            inferences AS (
                SELECT id, variant_name
                FROM tensorzero.chat_inferences
                WHERE function_name = $3 AND tags->>'tensorzero::namespace' = $4
                UNION ALL
                SELECT id, variant_name
                FROM tensorzero.json_inferences
                WHERE function_name = $5 AND tags->>'tensorzero::namespace' = $6
                UNION ALL
                SELECT DISTINCT episode_id AS id, variant_name
                FROM tensorzero.chat_inferences
                WHERE function_name = $7 AND tags->>'tensorzero::namespace' = $8 AND episode_id IS NOT NULL
                UNION ALL
                SELECT DISTINCT episode_id AS id, variant_name
                FROM tensorzero.json_inferences
                WHERE function_name = $9 AND tags->>'tensorzero::namespace' = $10 AND episode_id IS NOT NULL
            )
            SELECT
                i.variant_name,
                AVG(f.value)::REAL as mean,
                VAR_SAMP(f.value)::REAL as variance,
                COUNT(*)::BIGINT as count
            FROM feedback f
            JOIN inferences i ON f.target_id = i.id
            WHERE 1=1 GROUP BY i.variant_name
            ",
        );
    }

    #[test]
    fn test_build_feedback_by_variant_query_with_namespace_and_variants() {
        let variant_names = vec!["variant_a".to_string()];
        let qb = build_feedback_by_variant_query(
            "my_metric",
            "my_function",
            &variant_names,
            Some("staging"),
            None,
        );
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT target_id, value::INT::DOUBLE PRECISION as value
                FROM tensorzero.boolean_metric_feedback
                WHERE metric_name = $1
                UNION ALL
                SELECT target_id, value
                FROM tensorzero.float_metric_feedback
                WHERE metric_name = $2
            ),
            inferences AS (
                SELECT id, variant_name
                FROM tensorzero.chat_inferences
                WHERE function_name = $3 AND tags->>'tensorzero::namespace' = $4
                UNION ALL
                SELECT id, variant_name
                FROM tensorzero.json_inferences
                WHERE function_name = $5 AND tags->>'tensorzero::namespace' = $6
                UNION ALL
                SELECT DISTINCT episode_id AS id, variant_name
                FROM tensorzero.chat_inferences
                WHERE function_name = $7 AND tags->>'tensorzero::namespace' = $8 AND episode_id IS NOT NULL
                UNION ALL
                SELECT DISTINCT episode_id AS id, variant_name
                FROM tensorzero.json_inferences
                WHERE function_name = $9 AND tags->>'tensorzero::namespace' = $10 AND episode_id IS NOT NULL
            )
            SELECT
                i.variant_name,
                AVG(f.value)::REAL as mean,
                VAR_SAMP(f.value)::REAL as variance,
                COUNT(*)::BIGINT as count
            FROM feedback f
            JOIN inferences i ON f.target_id = i.id
            WHERE 1=1 AND i.variant_name = ANY($11) GROUP BY i.variant_name
            ",
        );
    }

    #[test]
    fn test_build_feedback_by_variant_query_with_max_samples() {
        let qb = build_feedback_by_variant_query("my_metric", "my_function", &[], None, Some(100));
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT target_id, value::INT::DOUBLE PRECISION as value
                FROM tensorzero.boolean_metric_feedback
                WHERE metric_name = $1
                UNION ALL
                SELECT target_id, value
                FROM tensorzero.float_metric_feedback
                WHERE metric_name = $2
            ),
            inferences AS (
                SELECT id, variant_name
                FROM tensorzero.chat_inferences
                WHERE function_name = $3
                UNION ALL
                SELECT id, variant_name
                FROM tensorzero.json_inferences
                WHERE function_name = $4
            ),
            numbered AS (
                SELECT
                    i.variant_name,
                    f.value,
                    ROW_NUMBER() OVER (PARTITION BY i.variant_name ORDER BY f.target_id DESC) as rn
                FROM feedback f
                JOIN inferences i ON f.target_id = i.id
                WHERE 1=1
            )
            SELECT
                variant_name,
                AVG(value)::REAL as mean,
                VAR_SAMP(value)::REAL as variance,
                COUNT(*)::BIGINT as count
            FROM numbered
            WHERE rn <= $5 GROUP BY variant_name
            ",
        );
    }

    #[test]
    fn test_build_feedback_by_variant_query_with_namespace_and_max_samples() {
        let variant_names = vec!["variant_a".to_string(), "variant_b".to_string()];
        let qb = build_feedback_by_variant_query(
            "my_metric",
            "my_function",
            &variant_names,
            Some("production"),
            Some(50),
        );
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT target_id, value::INT::DOUBLE PRECISION as value
                FROM tensorzero.boolean_metric_feedback
                WHERE metric_name = $1
                UNION ALL
                SELECT target_id, value
                FROM tensorzero.float_metric_feedback
                WHERE metric_name = $2
            ),
            inferences AS (
                SELECT id, variant_name
                FROM tensorzero.chat_inferences
                WHERE function_name = $3 AND tags->>'tensorzero::namespace' = $4
                UNION ALL
                SELECT id, variant_name
                FROM tensorzero.json_inferences
                WHERE function_name = $5 AND tags->>'tensorzero::namespace' = $6
                UNION ALL
                SELECT DISTINCT episode_id AS id, variant_name
                FROM tensorzero.chat_inferences
                WHERE function_name = $7 AND tags->>'tensorzero::namespace' = $8 AND episode_id IS NOT NULL
                UNION ALL
                SELECT DISTINCT episode_id AS id, variant_name
                FROM tensorzero.json_inferences
                WHERE function_name = $9 AND tags->>'tensorzero::namespace' = $10 AND episode_id IS NOT NULL
            ),
            numbered AS (
                SELECT
                    i.variant_name,
                    f.value,
                    ROW_NUMBER() OVER (PARTITION BY i.variant_name ORDER BY f.target_id DESC) as rn
                FROM feedback f
                JOIN inferences i ON f.target_id = i.id
                WHERE 1=1 AND i.variant_name = ANY($11)
            )
            SELECT
                variant_name,
                AVG(value)::REAL as mean,
                VAR_SAMP(value)::REAL as variance,
                COUNT(*)::BIGINT as count
            FROM numbered
            WHERE rn <= $12 GROUP BY variant_name
            ",
        );
    }

    // ===== build_metrics_with_feedback_query tests =====

    #[test]
    fn test_build_metrics_with_feedback_query_chat_no_variant() {
        let qb =
            build_metrics_with_feedback_query("my_function", "tensorzero.chat_inferences", None);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH inference_ids AS (SELECT id FROM tensorzero.chat_inferences WHERE function_name = $1)
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
            ORDER BY metric_name
            ",
        );
    }

    #[test]
    fn test_build_metrics_with_feedback_query_chat_with_variant() {
        let qb = build_metrics_with_feedback_query(
            "my_function",
            "tensorzero.chat_inferences",
            Some("my_variant"),
        );
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH inference_ids AS (SELECT id FROM tensorzero.chat_inferences
                WHERE function_name = $1 AND variant_name = $2)
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
            ORDER BY metric_name
            ",
        );
    }

    #[test]
    fn test_build_metrics_with_feedback_query_json_table() {
        let qb =
            build_metrics_with_feedback_query("my_function", "tensorzero.json_inferences", None);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH inference_ids AS (SELECT id FROM tensorzero.json_inferences WHERE function_name = $1)
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
            ORDER BY metric_name
            ",
        );
    }

    // ===== build_cumulative_feedback_timeseries_query tests =====

    #[test]
    fn test_build_cumulative_feedback_timeseries_query_day_no_variants() {
        let qb =
            build_cumulative_feedback_timeseries_query("my_metric", "my_function", &[], "day", 30);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback_with_variant AS (SELECT date_trunc('day', f.created_at) + INTERVAL '1 day' AS period_end,
                i.variant_name,
                f.value
            FROM (
                SELECT target_id, value::INT::DOUBLE PRECISION as value, created_at
                FROM tensorzero.boolean_metric_feedback WHERE metric_name = $1
                UNION ALL
                SELECT target_id, value, created_at
                FROM tensorzero.float_metric_feedback WHERE metric_name = $2
            ) f
            JOIN (
                SELECT id, variant_name FROM tensorzero.chat_inferences WHERE function_name = $3
                UNION ALL
                SELECT id, variant_name FROM tensorzero.json_inferences WHERE function_name = $4) i ON f.target_id = i.id WHERE 1=1
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
            WHERE period_end >= (SELECT max_end FROM max_period) - ($5 || ' day')::INTERVAL ORDER BY period_end, variant_name
            ",
        );
    }

    #[test]
    fn test_build_cumulative_feedback_timeseries_query_hour_with_variants() {
        let variant_names = vec!["variant_a".to_string(), "variant_b".to_string()];
        let qb = build_cumulative_feedback_timeseries_query(
            "my_metric",
            "my_function",
            &variant_names,
            "hour",
            24,
        );
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback_with_variant AS (SELECT date_trunc('hour', f.created_at) + INTERVAL '1 hour' AS period_end,
                i.variant_name,
                f.value
            FROM (
                SELECT target_id, value::INT::DOUBLE PRECISION as value, created_at
                FROM tensorzero.boolean_metric_feedback WHERE metric_name = $1
                UNION ALL
                SELECT target_id, value, created_at
                FROM tensorzero.float_metric_feedback WHERE metric_name = $2
            ) f
            JOIN (
                SELECT id, variant_name FROM tensorzero.chat_inferences WHERE function_name = $3
                UNION ALL
                SELECT id, variant_name FROM tensorzero.json_inferences WHERE function_name = $4) i ON f.target_id = i.id WHERE 1=1 AND i.variant_name = ANY($5)
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
            WHERE period_end >= (SELECT max_end FROM max_period) - ($6 || ' hour')::INTERVAL ORDER BY period_end, variant_name
            ",
        );
    }

    #[test]
    fn test_build_cumulative_feedback_timeseries_query_week() {
        let qb =
            build_cumulative_feedback_timeseries_query("my_metric", "my_function", &[], "week", 12);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        // Just check that week interval is used correctly
        assert!(
            sql.contains("date_trunc('week'"),
            "Query should use week truncation"
        );
        assert!(
            sql.contains("INTERVAL '1 week'"),
            "Query should use week interval"
        );
        assert!(
            sql.contains("' week')::INTERVAL"),
            "Query should use week in period filter"
        );
    }

    // ===== build_variant_performances_query tests =====

    #[test]
    fn test_build_variant_performances_query_inference_level_boolean() {
        let params = VariantPerformancesQueryParams {
            metric_name: "my_metric",
            function_name: "my_function",
            inference_table: "tensorzero.chat_inferences",
            metric_table: "tensorzero.boolean_metric_feedback",
            value_cast: "value::INT::DOUBLE PRECISION",
            time_bucket_expr: "date_trunc('day', i.created_at)",
            metric_level: MetricConfigLevel::Inference,
            variant_name: None,
        };
        let qb = build_variant_performances_query(&params);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT DISTINCT ON (target_id) target_id, value::INT::DOUBLE PRECISION as value, created_at
                FROM tensorzero.boolean_metric_feedback WHERE metric_name = $1 ORDER BY target_id, created_at DESC
            )
            SELECT date_trunc('day', i.created_at) as period_start,
                i.variant_name,
                COUNT(*)::INT as count,
                AVG(f.value) as avg_metric,
                STDDEV_SAMP(f.value) as stdev,
                CASE WHEN COUNT(*) >= 2 THEN 1.96 * (STDDEV_SAMP(f.value) / SQRT(COUNT(*))) END as ci_error
            FROM tensorzero.chat_inferences i
            JOIN feedback f ON f.target_id = i.id
            WHERE i.function_name = $2 GROUP BY period_start, i.variant_name ORDER BY period_start ASC, i.variant_name ASC
            ",
        );
    }

    #[test]
    fn test_build_variant_performances_query_inference_level_float() {
        let params = VariantPerformancesQueryParams {
            metric_name: "my_metric",
            function_name: "my_function",
            inference_table: "tensorzero.json_inferences",
            metric_table: "tensorzero.float_metric_feedback",
            value_cast: "value::DOUBLE PRECISION",
            time_bucket_expr: "date_trunc('hour', i.created_at)",
            metric_level: MetricConfigLevel::Inference,
            variant_name: None,
        };
        let qb = build_variant_performances_query(&params);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT DISTINCT ON (target_id) target_id, value::DOUBLE PRECISION as value, created_at
                FROM tensorzero.float_metric_feedback WHERE metric_name = $1 ORDER BY target_id, created_at DESC
            )
            SELECT date_trunc('hour', i.created_at) as period_start,
                i.variant_name,
                COUNT(*)::INT as count,
                AVG(f.value) as avg_metric,
                STDDEV_SAMP(f.value) as stdev,
                CASE WHEN COUNT(*) >= 2 THEN 1.96 * (STDDEV_SAMP(f.value) / SQRT(COUNT(*))) END as ci_error
            FROM tensorzero.json_inferences i
            JOIN feedback f ON f.target_id = i.id
            WHERE i.function_name = $2 GROUP BY period_start, i.variant_name ORDER BY period_start ASC, i.variant_name ASC
            ",
        );
    }

    #[test]
    fn test_build_variant_performances_query_inference_level_with_variant() {
        let params = VariantPerformancesQueryParams {
            metric_name: "my_metric",
            function_name: "my_function",
            inference_table: "tensorzero.chat_inferences",
            metric_table: "tensorzero.boolean_metric_feedback",
            value_cast: "value::INT::DOUBLE PRECISION",
            time_bucket_expr: "date_trunc('day', i.created_at)",
            metric_level: MetricConfigLevel::Inference,
            variant_name: Some("my_variant"),
        };
        let qb = build_variant_performances_query(&params);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT DISTINCT ON (target_id) target_id, value::INT::DOUBLE PRECISION as value, created_at
                FROM tensorzero.boolean_metric_feedback WHERE metric_name = $1 ORDER BY target_id, created_at DESC
            )
            SELECT date_trunc('day', i.created_at) as period_start,
                i.variant_name,
                COUNT(*)::INT as count,
                AVG(f.value) as avg_metric,
                STDDEV_SAMP(f.value) as stdev,
                CASE WHEN COUNT(*) >= 2 THEN 1.96 * (STDDEV_SAMP(f.value) / SQRT(COUNT(*))) END as ci_error
            FROM tensorzero.chat_inferences i
            JOIN feedback f ON f.target_id = i.id
            WHERE i.function_name = $2 AND i.variant_name = $3 GROUP BY period_start, i.variant_name ORDER BY period_start ASC, i.variant_name ASC
            ",
        );
    }

    #[test]
    fn test_build_variant_performances_query_episode_level_boolean() {
        let params = VariantPerformancesQueryParams {
            metric_name: "my_metric",
            function_name: "my_function",
            inference_table: "tensorzero.chat_inferences",
            metric_table: "tensorzero.boolean_metric_feedback",
            value_cast: "value::INT::DOUBLE PRECISION",
            time_bucket_expr: "date_trunc('day', i.created_at)",
            metric_level: MetricConfigLevel::Episode,
            variant_name: None,
        };
        let qb = build_variant_performances_query(&params);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT DISTINCT ON (target_id) target_id, value::INT::DOUBLE PRECISION as value
                FROM tensorzero.boolean_metric_feedback WHERE metric_name = $1 ORDER BY target_id, created_at DESC
            ),
            per_episode AS (
                SELECT date_trunc('day', i.created_at) AS period_start,
                    i.variant_name,
                    i.episode_id,
                    f.value
                FROM tensorzero.chat_inferences i
                JOIN feedback f ON f.target_id = i.episode_id
                WHERE i.function_name = $2 GROUP BY period_start, i.variant_name, i.episode_id, f.value
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
            ORDER BY period_start ASC, variant_name ASC
            ",
        );
    }

    #[test]
    fn test_build_variant_performances_query_episode_level_with_variant() {
        let params = VariantPerformancesQueryParams {
            metric_name: "my_metric",
            function_name: "my_function",
            inference_table: "tensorzero.chat_inferences",
            metric_table: "tensorzero.boolean_metric_feedback",
            value_cast: "value::INT::DOUBLE PRECISION",
            time_bucket_expr: "date_trunc('day', i.created_at)",
            metric_level: MetricConfigLevel::Episode,
            variant_name: Some("my_variant"),
        };
        let qb = build_variant_performances_query(&params);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH feedback AS (
                SELECT DISTINCT ON (target_id) target_id, value::INT::DOUBLE PRECISION as value
                FROM tensorzero.boolean_metric_feedback WHERE metric_name = $1 ORDER BY target_id, created_at DESC
            ),
            per_episode AS (
                SELECT date_trunc('day', i.created_at) AS period_start,
                    i.variant_name,
                    i.episode_id,
                    f.value
                FROM tensorzero.chat_inferences i
                JOIN feedback f ON f.target_id = i.episode_id
                WHERE i.function_name = $2 AND i.variant_name = $3 GROUP BY period_start, i.variant_name, i.episode_id, f.value
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
            ORDER BY period_start ASC, variant_name ASC
            ",
        );
    }

    #[test]
    fn test_build_variant_performances_query_cumulative_time_window() {
        let params = VariantPerformancesQueryParams {
            metric_name: "my_metric",
            function_name: "my_function",
            inference_table: "tensorzero.chat_inferences",
            metric_table: "tensorzero.boolean_metric_feedback",
            value_cast: "value::INT::DOUBLE PRECISION",
            time_bucket_expr: "'1970-01-01 00:00:00'::TIMESTAMPTZ",
            metric_level: MetricConfigLevel::Inference,
            variant_name: None,
        };
        let qb = build_variant_performances_query(&params);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        // Verify the cumulative time bucket is used
        assert!(
            sql.contains("'1970-01-01 00:00:00'::TIMESTAMPTZ as period_start"),
            "Query should use cumulative timestamp"
        );
    }
}
