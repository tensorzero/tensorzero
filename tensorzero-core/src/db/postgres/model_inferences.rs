//! Model inference queries for Postgres.
//!
//! This module implements read and write operations for the model_inferences table in Postgres.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lazy_static::lazy_static;
use sqlx::types::Json;
use sqlx::{PgPool, QueryBuilder, Row};
use std::collections::BTreeMap;
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::db::model_inferences::ModelInferenceQueries;
use crate::db::query_helpers::uuid_to_datetime;
use crate::db::{ModelLatencyDatapoint, ModelUsageTimePoint, TimeWindow};
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::inference::types::{
    ContentBlockOutput, FinishReason, StoredModelInference, StoredRequestMessage,
};

use super::PostgresConnectionInfo;

/// Quantiles used for Postgres latency queries.
pub const POSTGRES_QUANTILES: &[f64; 17] = &[
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995,
    0.999,
];

const LATENCY_HISTOGRAM_BUCKETS_PER_POWER_OF_TWO: f64 = 64.0;
const RESPONSE_TIME_MS_METRIC: &str = "response_time_ms";
const TTFT_MS_METRIC: &str = "ttft_ms";

#[derive(Debug, Clone, Copy)]
struct LatencyHistogramBucket {
    bucket_id: i32,
    bucket_count: f64,
}

#[derive(Debug, sqlx::FromRow)]
struct ModelLatencyHistogramBucketRow {
    model_name: String,
    count: i64,
    metric: Option<String>,
    bucket_id: Option<i32>,
    bucket_count: Option<f64>,
}

#[derive(Debug, Default)]
struct ModelLatencyHistogramModelData {
    count: u64,
    response_time_ms_buckets: Vec<LatencyHistogramBucket>,
    ttft_ms_buckets: Vec<LatencyHistogramBucket>,
}
lazy_static! {
    /// Quantiles array string for Postgres queries.
    pub static ref POSTGRES_QUANTILES_ARRAY_STRING: String = POSTGRES_QUANTILES
        .to_vec()
        .iter()
        .map(|q| q.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    pub static ref EMPTY_QUANTILES: Vec<Option<f32>> = vec![None; POSTGRES_QUANTILES.len()];
}

// =====================================================================
// ModelInferenceQueries trait implementation
// =====================================================================

#[async_trait]
impl ModelInferenceQueries for PostgresConnectionInfo {
    async fn get_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<StoredModelInference>, Error> {
        let pool = self.get_pool_result()?;

        let mut qb = build_get_model_inferences_query(inference_id);
        let rows: Vec<StoredModelInference> = qb.build_query_as().fetch_all(pool).await?;

        Ok(rows)
    }

    async fn insert_model_inferences(&self, rows: &[StoredModelInference]) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        if let Some(batch_sender) = self.batch_sender() {
            return batch_sender.send_model_inferences(rows);
        }

        let pool = self.get_pool_result()?;
        let mut metadata_qb = build_insert_model_inferences_query(rows)?;
        metadata_qb.build().execute(pool).await?;
        let mut io_qb = build_insert_model_inference_data_query(rows)?;
        io_qb.build().execute(pool).await?;
        Ok(())
    }

    async fn count_distinct_models_used(&self) -> Result<u32, Error> {
        let pool = self.get_pool_result()?;
        count_distinct_models_used_impl(pool).await
    }

    async fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<ModelUsageTimePoint>, Error> {
        let pool = self.get_pool_result()?;
        get_model_usage_timeseries_impl(pool, time_window, max_periods).await
    }

    async fn get_model_latency_quantiles(
        &self,
        time_window: TimeWindow,
    ) -> Result<Vec<ModelLatencyDatapoint>, Error> {
        let pool = self.get_pool_result()?;

        if time_window == TimeWindow::Minute {
            let mut query_builder = build_model_latency_quantiles_raw_query(&time_window);
            let rows: Vec<ModelLatencyDatapoint> =
                query_builder.build_query_as().fetch_all(pool).await?;
            return Ok(rows);
        }

        let (source_table, source_time_column, time_window_interval) = match time_window {
            TimeWindow::Hour => (
                "tensorzero.model_latency_histogram_minute",
                "minute",
                Some("1 hour"),
            ),
            TimeWindow::Day => (
                "tensorzero.model_latency_histogram_hour",
                "hour",
                Some("1 day"),
            ),
            TimeWindow::Week => (
                "tensorzero.model_latency_histogram_hour",
                "hour",
                Some("1 week"),
            ),
            TimeWindow::Month => (
                "tensorzero.model_latency_histogram_hour",
                "hour",
                Some("1 month"),
            ),
            TimeWindow::Cumulative => ("tensorzero.model_latency_histogram_hour", "hour", None),
            TimeWindow::Minute => {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: format!(
                        "Trying to handle TimeWindow::Minute when constructing histogram query. {IMPOSSIBLE_ERROR_MESSAGE}"
                    ),
                }));
            }
        };

        let mut query_builder = build_model_latency_quantiles_histogram_query(
            source_table,
            source_time_column,
            time_window_interval,
        );
        let bucket_rows: Vec<ModelLatencyHistogramBucketRow> =
            query_builder.build_query_as().fetch_all(pool).await?;

        Ok(compute_model_latency_quantiles_from_histogram_buckets(
            bucket_rows,
        ))
    }

    fn get_model_latency_quantile_function_inputs(&self) -> &[f64] {
        POSTGRES_QUANTILES
    }
}

// =====================================================================
// Query builder functions (for unit testing)
// =====================================================================

/// Builds a query to get model inferences by inference_id.
fn build_get_model_inferences_query(inference_id: Uuid) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            i.id,
            i.inference_id,
            io.raw_request,
            io.raw_response,
            io.system,
            io.input_messages,
            io.output,
            i.input_tokens,
            i.output_tokens,
            i.response_time_ms,
            i.model_name,
            i.model_provider_name,
            i.ttft_ms,
            i.cached,
            i.finish_reason,
            i.snapshot_hash,
            i.created_at
        FROM tensorzero.model_inferences i
        LEFT JOIN tensorzero.model_inference_data io ON io.id = i.id AND io.created_at = i.created_at
        WHERE i.inference_id = ",
    );
    qb.push_bind(inference_id);

    qb
}

/// Builds a query to insert model inference metadata.
pub(super) fn build_insert_model_inferences_query(
    rows: &[StoredModelInference],
) -> Result<QueryBuilder<sqlx::Postgres>, Error> {
    // Pre-compute timestamps from UUIDs to propagate errors before entering push_values
    let timestamps: Vec<DateTime<Utc>> = rows
        .iter()
        .map(|row| uuid_to_datetime(row.id))
        .collect::<Result<_, _>>()?;

    let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.model_inferences (
            id, inference_id, input_tokens, output_tokens,
            response_time_ms, model_name, model_provider_name,
            ttft_ms, cached, finish_reason, snapshot_hash, created_at
        ) ",
    );

    qb.push_values(rows.iter().zip(&timestamps), |mut b, (row, created_at)| {
        let snapshot_hash_bytes: Option<Vec<u8>> =
            row.snapshot_hash.as_ref().map(|h| h.as_bytes().to_vec());

        b.push_bind(row.id)
            .push_bind(row.inference_id)
            .push_bind(row.input_tokens.map(|v| v as i32))
            .push_bind(row.output_tokens.map(|v| v as i32))
            .push_bind(row.response_time_ms.map(|v| v as i32))
            .push_bind(&row.model_name)
            .push_bind(&row.model_provider_name)
            .push_bind(row.ttft_ms.map(|v| v as i32))
            .push_bind(row.cached)
            .push_bind(row.finish_reason)
            .push_bind(snapshot_hash_bytes)
            .push_bind(created_at);
    });

    Ok(qb)
}

/// Builds a query to insert model inference IO data.
pub(super) fn build_insert_model_inference_data_query(
    rows: &[StoredModelInference],
) -> Result<QueryBuilder<sqlx::Postgres>, Error> {
    let timestamps: Vec<DateTime<Utc>> = rows
        .iter()
        .map(|row| uuid_to_datetime(row.id))
        .collect::<Result<_, _>>()?;

    let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.model_inference_data (
            id, raw_request, raw_response, system,
            input_messages, output, created_at
        ) ",
    );

    qb.push_values(rows.iter().zip(&timestamps), |mut b, (row, created_at)| {
        b.push_bind(row.id)
            .push_bind(&row.raw_request)
            .push_bind(&row.raw_response)
            .push_bind(&row.system)
            .push_bind(row.input_messages.as_ref().map(Json::from))
            .push_bind(row.output.as_ref().map(Json::from))
            .push_bind(created_at);
    });

    Ok(qb)
}

// =====================================================================
// Model statistics query implementations
// =====================================================================

async fn count_distinct_models_used_impl(pool: &PgPool) -> Result<u32, Error> {
    let row: (i64,) = sqlx::query_as(
        r"
        SELECT COUNT(DISTINCT model_name)
        FROM tensorzero.model_provider_statistics
        ",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to count distinct models: {e}"),
        })
    })?;

    Ok(row.0 as u32)
}

async fn get_model_usage_timeseries_impl(
    pool: &PgPool,
    time_window: TimeWindow,
    max_periods: u32,
) -> Result<Vec<ModelUsageTimePoint>, Error> {
    // For cumulative, we aggregate everything into a single period at epoch
    if time_window == TimeWindow::Cumulative {
        return get_model_usage_cumulative(pool).await;
    }

    let mut query_builder = build_model_usage_timeseries_query(&time_window, max_periods);
    let rows: Vec<ModelUsageTimePoint> = query_builder.build_query_as().fetch_all(pool).await?;

    Ok(rows)
}

/// Builds the query for model usage timeseries (non-cumulative).
fn build_model_usage_timeseries_query(
    time_window: &TimeWindow,
    max_periods: u32,
) -> QueryBuilder<sqlx::Postgres> {
    let time_unit = time_window.to_postgres_time_unit();

    // Build the query dynamically since date_trunc requires a literal string
    // and we can't bind it as a parameter
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("SELECT date_trunc('");
    query_builder.push(time_unit);
    query_builder.push(
        "', minute) as period_start,
            model_name,
            SUM(total_input_tokens)::BIGINT as input_tokens,
            SUM(total_output_tokens)::BIGINT as output_tokens,
            SUM(inference_count)::BIGINT as count
        FROM tensorzero.model_provider_statistics
        WHERE minute >= (
            SELECT COALESCE(MAX(date_trunc('",
    );
    query_builder.push(time_unit);
    query_builder.push(
        "', minute)), '1970-01-01'::TIMESTAMPTZ)
            FROM tensorzero.model_provider_statistics
        ) - INTERVAL '",
    );
    query_builder.push(max_periods.to_string());
    query_builder.push(" ");
    query_builder.push(time_unit);
    query_builder.push(
        "s'
        GROUP BY period_start, model_name
        ORDER BY period_start DESC, model_name",
    );

    query_builder
}

async fn get_model_usage_cumulative(pool: &PgPool) -> Result<Vec<ModelUsageTimePoint>, Error> {
    let rows: Vec<ModelUsageTimePoint> = sqlx::query_as(
        r"
        SELECT
            '1970-01-01'::TIMESTAMPTZ as period_start,
            model_name,
            SUM(total_input_tokens)::BIGINT as input_tokens,
            SUM(total_output_tokens)::BIGINT as output_tokens,
            SUM(inference_count)::BIGINT as count
        FROM tensorzero.model_provider_statistics
        GROUP BY model_name
        ORDER BY model_name
        ",
    )
    .fetch_all(pool)
    .await?;

    Ok(rows)
}

/// Builds a query that returns aggregated histogram buckets.
fn build_model_latency_quantiles_histogram_query(
    source_table: &str,
    source_time_column: &str,
    time_window: Option<&str>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        "WITH model_counts AS (
            SELECT
                model_name,
                SUM(inference_count)::BIGINT AS count
            FROM tensorzero.model_provider_statistics
            WHERE TRUE",
    );

    push_now_window_filter(&mut query_builder, "minute", time_window);
    query_builder.push(
        "
            GROUP BY model_name
        ),
        hist AS (
            SELECT
                model_name,
                metric,
                bucket_id,
                SUM(bucket_count)::DOUBLE PRECISION AS bucket_count
            FROM ",
    );
    query_builder.push(source_table);
    query_builder.push(
        "
            WHERE TRUE",
    );
    push_now_window_filter(&mut query_builder, source_time_column, time_window);
    query_builder.push(
        "
            GROUP BY model_name, metric, bucket_id
        )
        SELECT
            mc.model_name,
            mc.count,
            h.metric,
            h.bucket_id,
            h.bucket_count
        FROM model_counts mc
        LEFT JOIN hist h
          ON h.model_name = mc.model_name
        ORDER BY mc.model_name, h.metric, h.bucket_id",
    );

    query_builder
}

/// Pushes an optional window filter anchored to `NOW()`.
fn push_now_window_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    time_column: &str,
    time_window: Option<&str>,
) {
    if let Some(time_window) = time_window {
        query_builder.push(" AND ");
        query_builder.push(time_column);
        query_builder.push(" >= NOW() - INTERVAL '");
        query_builder.push(time_window);
        query_builder.push("' AND ");
        query_builder.push(time_column);
        query_builder.push(" <= NOW()");
    }
}

fn compute_model_latency_quantiles_from_histogram_buckets(
    rows: Vec<ModelLatencyHistogramBucketRow>,
) -> Vec<ModelLatencyDatapoint> {
    let mut model_data_by_name: BTreeMap<String, ModelLatencyHistogramModelData> = BTreeMap::new();

    for row in rows {
        let model_data = model_data_by_name.entry(row.model_name).or_default();
        model_data.count = row.count.max(0) as u64;

        let (metric, bucket_id, bucket_count) = match (row.metric, row.bucket_id, row.bucket_count)
        {
            (Some(metric), Some(bucket_id), Some(bucket_count)) => {
                (metric, bucket_id, bucket_count)
            }
            _ => continue,
        };

        let bucket = LatencyHistogramBucket {
            bucket_id,
            bucket_count,
        };

        match metric.as_str() {
            RESPONSE_TIME_MS_METRIC => model_data.response_time_ms_buckets.push(bucket),
            TTFT_MS_METRIC => model_data.ttft_ms_buckets.push(bucket),
            _ => {}
        }
    }

    model_data_by_name
        .into_iter()
        .map(|(model_name, model_data)| ModelLatencyDatapoint {
            model_name,
            response_time_ms_quantiles: compute_quantiles_from_histogram_buckets(
                &model_data.response_time_ms_buckets,
            ),
            ttft_ms_quantiles: compute_quantiles_from_histogram_buckets(
                &model_data.ttft_ms_buckets,
            ),
            count: model_data.count,
        })
        .collect()
}

fn compute_quantiles_from_histogram_buckets(
    buckets: &[LatencyHistogramBucket],
) -> Vec<Option<f32>> {
    if buckets.is_empty() {
        return EMPTY_QUANTILES.clone();
    }

    let mut sorted_buckets = buckets.to_vec();
    sorted_buckets.sort_by_key(|bucket| bucket.bucket_id);
    let sample_count: f64 = sorted_buckets
        .iter()
        .map(|bucket| bucket.bucket_count)
        .sum();

    if sample_count <= 0.0 {
        return EMPTY_QUANTILES.clone();
    }

    // TODO(shuyangli): Switch this to do a single pass instead of O(n^2).
    POSTGRES_QUANTILES
        .iter()
        .map(|quantile| {
            compute_quantile_from_histogram_buckets(&sorted_buckets, *quantile, sample_count)
        })
        .collect()
}

fn compute_quantile_from_histogram_buckets(
    sorted_buckets: &[LatencyHistogramBucket],
    quantile: f64,
    sample_count: f64,
) -> Option<f32> {
    let rank_target = 1.0 + quantile * (sample_count - 1.0);
    let mut cumulative_count = 0.0;

    for bucket in sorted_buckets {
        cumulative_count += bucket.bucket_count;

        if cumulative_count < rank_target {
            continue;
        }

        if bucket.bucket_count <= 0.0 {
            return None;
        }

        let lower_ms = latency_bucket_lower_ms(bucket.bucket_id);
        let upper_ms = latency_bucket_upper_ms(bucket.bucket_id);

        let quantile_ms = if lower_ms <= 0.0 || upper_ms <= 0.0 {
            0.0
        } else {
            (lower_ms * upper_ms).sqrt()
        };

        return Some(quantile_ms as f32);
    }

    None
}

fn latency_bucket_lower_ms(bucket_id: i32) -> f64 {
    if bucket_id < 0 {
        0.0
    } else {
        2f64.powf(bucket_id as f64 / LATENCY_HISTOGRAM_BUCKETS_PER_POWER_OF_TWO)
    }
}

fn latency_bucket_upper_ms(bucket_id: i32) -> f64 {
    if bucket_id < 0 {
        1.0
    } else {
        2f64.powf((bucket_id + 1) as f64 / LATENCY_HISTOGRAM_BUCKETS_PER_POWER_OF_TWO)
    }
}

/// Builds a query to compute latency quantiles from raw model_inferences data.
fn build_model_latency_quantiles_raw_query(
    time_window: &TimeWindow,
) -> QueryBuilder<sqlx::Postgres> {
    let num_quantiles = POSTGRES_QUANTILES.len();
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        "SELECT
            model_name,
            COALESCE(
                percentile_cont(ARRAY[",
    );
    query_builder.push(&*POSTGRES_QUANTILES_ARRAY_STRING);
    query_builder.push(
        "]) WITHIN GROUP (ORDER BY response_time_ms),
                array_fill(NULL::double precision, ARRAY[",
    );
    query_builder.push(num_quantiles.to_string());
    query_builder.push(
        "])
            ) AS response_time_ms_quantiles,
            COALESCE(
                percentile_cont(ARRAY[",
    );
    query_builder.push(&*POSTGRES_QUANTILES_ARRAY_STRING);
    query_builder.push(
        "]) WITHIN GROUP (ORDER BY ttft_ms),
                array_fill(NULL::double precision, ARRAY[",
    );
    query_builder.push(num_quantiles.to_string());
    query_builder.push(
        "])
            ) AS ttft_ms_quantiles,
            COUNT(*)::BIGINT as count
        FROM tensorzero.model_inferences
        WHERE ",
    );

    // Add time filter based on time_window
    match time_window {
        TimeWindow::Minute => query_builder.push("created_at >= NOW() - INTERVAL '1 minute'"),
        TimeWindow::Hour => query_builder.push("created_at >= NOW() - INTERVAL '1 hour'"),
        TimeWindow::Day => query_builder.push("created_at >= NOW() - INTERVAL '1 day'"),
        TimeWindow::Week => query_builder.push("created_at >= NOW() - INTERVAL '1 week'"),
        TimeWindow::Month => query_builder.push("created_at >= NOW() - INTERVAL '1 month'"),
        TimeWindow::Cumulative => query_builder.push("TRUE"),
    };

    query_builder.push(
        "
        GROUP BY model_name
        ORDER BY model_name",
    );

    query_builder
}

// =====================================================================
// FromRow implementations
// =====================================================================

impl sqlx::FromRow<'_, sqlx::postgres::PgRow> for ModelLatencyDatapoint {
    fn from_row(row: &sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let model_name: String = row.try_get("model_name")?;
        // Read as f64 (double precision) and convert to f32
        let response_time_ms_quantiles: Vec<Option<f32>> = row
            .try_get::<Option<Vec<Option<f64>>>, _>("response_time_ms_quantiles")?
            .map(|v| v.into_iter().map(|x| x.map(|x| x as f32)).collect())
            .unwrap_or_else(|| EMPTY_QUANTILES.clone());
        let ttft_ms_quantiles: Vec<Option<f32>> = row
            .try_get::<Option<Vec<Option<f64>>>, _>("ttft_ms_quantiles")?
            .map(|v| v.into_iter().map(|x| x.map(|x| x as f32)).collect())
            .unwrap_or_else(|| EMPTY_QUANTILES.clone());
        let count: i64 = row.try_get("count")?;

        Ok(Self {
            model_name,
            response_time_ms_quantiles,
            ttft_ms_quantiles,
            count: count as u64,
        })
    }
}

impl sqlx::FromRow<'_, sqlx::postgres::PgRow> for ModelUsageTimePoint {
    fn from_row(row: &sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let period_start: DateTime<Utc> = row.try_get("period_start")?;
        let input_tokens: Option<i64> = row.try_get("input_tokens")?;
        let output_tokens: Option<i64> = row.try_get("output_tokens")?;
        let count: Option<i64> = row.try_get("count")?;

        Ok(ModelUsageTimePoint {
            period_start,
            model_name: row.try_get("model_name")?,
            input_tokens: input_tokens.map(|v| v as u64),
            output_tokens: output_tokens.map(|v| v as u64),
            count: count.map(|v| v as u64),
        })
    }
}

/// Manual implementation of FromRow for StoredModelInference.
/// This allows direct deserialization from Postgres rows.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for StoredModelInference {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let id: Uuid = row.try_get("id")?;
        let inference_id: Uuid = row.try_get("inference_id")?;
        let raw_request: Option<String> = row.try_get("raw_request")?;
        let raw_response: Option<String> = row.try_get("raw_response")?;
        let system: Option<String> = row.try_get("system")?;
        let input_messages: Option<Json<Vec<StoredRequestMessage>>> =
            row.try_get("input_messages")?;
        let output: Option<Json<Vec<ContentBlockOutput>>> = row.try_get("output")?;
        let input_tokens: Option<i32> = row.try_get("input_tokens")?;
        let output_tokens: Option<i32> = row.try_get("output_tokens")?;
        let response_time_ms: Option<i32> = row.try_get("response_time_ms")?;
        let model_name: String = row.try_get("model_name")?;
        let model_provider_name: String = row.try_get("model_provider_name")?;
        let ttft_ms: Option<i32> = row.try_get("ttft_ms")?;
        let cached: bool = row.try_get("cached")?;
        let finish_reason: Option<FinishReason> = row.try_get("finish_reason")?;
        let snapshot_hash_bytes: Option<Vec<u8>> = row.try_get("snapshot_hash")?;
        let created_at: DateTime<Utc> = row.try_get("created_at")?;

        // Convert snapshot_hash from bytes
        let snapshot_hash = snapshot_hash_bytes.map(|bytes| SnapshotHash::from_bytes(&bytes));

        Ok(StoredModelInference {
            id,
            inference_id,
            raw_request,
            raw_response,
            system,
            input_messages: input_messages.map(|v| v.0),
            output: output.map(|v| v.0),
            input_tokens: input_tokens.map(|v| v as u32),
            output_tokens: output_tokens.map(|v| v as u32),
            response_time_ms: response_time_ms.map(|v| v as u32),
            model_name,
            model_provider_name,
            ttft_ms: ttft_ms.map(|v| v as u32),
            cached,
            finish_reason,
            snapshot_hash,
            timestamp: Some(created_at.to_rfc3339()),
        })
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_helpers::{
        assert_query_contains, assert_query_does_not_contain, assert_query_equals,
    };

    // =========================================================================
    // Model usage timeseries query tests
    // =========================================================================

    #[test]
    fn test_build_model_usage_timeseries_query_hour() {
        let qb = build_model_usage_timeseries_query(&TimeWindow::Hour, 24);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT date_trunc('hour', minute) as period_start,
            model_name,
            SUM(total_input_tokens)::BIGINT as input_tokens,
            SUM(total_output_tokens)::BIGINT as output_tokens,
            SUM(inference_count)::BIGINT as count
        FROM tensorzero.model_provider_statistics
        WHERE minute >= (
            SELECT COALESCE(MAX(date_trunc('hour', minute)), '1970-01-01'::TIMESTAMPTZ)
            FROM tensorzero.model_provider_statistics
        ) - INTERVAL '24 hours'
        GROUP BY period_start, model_name
        ORDER BY period_start DESC, model_name",
        );
    }

    #[test]
    fn test_build_model_usage_timeseries_query_day() {
        let qb = build_model_usage_timeseries_query(&TimeWindow::Day, 7);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT date_trunc('day', minute) as period_start,
            model_name,
            SUM(total_input_tokens)::BIGINT as input_tokens,
            SUM(total_output_tokens)::BIGINT as output_tokens,
            SUM(inference_count)::BIGINT as count
        FROM tensorzero.model_provider_statistics
        WHERE minute >= (
            SELECT COALESCE(MAX(date_trunc('day', minute)), '1970-01-01'::TIMESTAMPTZ)
            FROM tensorzero.model_provider_statistics
        ) - INTERVAL '7 days'
        GROUP BY period_start, model_name
        ORDER BY period_start DESC, model_name",
        );
    }

    #[test]
    fn test_build_model_usage_timeseries_query_minute() {
        let qb = build_model_usage_timeseries_query(&TimeWindow::Minute, 60);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT date_trunc('minute', minute) as period_start,
            model_name,
            SUM(total_input_tokens)::BIGINT as input_tokens,
            SUM(total_output_tokens)::BIGINT as output_tokens,
            SUM(inference_count)::BIGINT as count
        FROM tensorzero.model_provider_statistics
        WHERE minute >= (
            SELECT COALESCE(MAX(date_trunc('minute', minute)), '1970-01-01'::TIMESTAMPTZ)
            FROM tensorzero.model_provider_statistics
        ) - INTERVAL '60 minutes'
        GROUP BY period_start, model_name
        ORDER BY period_start DESC, model_name",
        );
    }

    #[test]
    fn test_build_model_usage_timeseries_query_week() {
        let qb = build_model_usage_timeseries_query(&TimeWindow::Week, 4);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT date_trunc('week', minute) as period_start,
            model_name,
            SUM(total_input_tokens)::BIGINT as input_tokens,
            SUM(total_output_tokens)::BIGINT as output_tokens,
            SUM(inference_count)::BIGINT as count
        FROM tensorzero.model_provider_statistics
        WHERE minute >= (
            SELECT COALESCE(MAX(date_trunc('week', minute)), '1970-01-01'::TIMESTAMPTZ)
            FROM tensorzero.model_provider_statistics
        ) - INTERVAL '4 weeks'
        GROUP BY period_start, model_name
        ORDER BY period_start DESC, model_name",
        );
    }

    #[test]
    fn test_build_model_usage_timeseries_query_month() {
        let qb = build_model_usage_timeseries_query(&TimeWindow::Month, 12);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT date_trunc('month', minute) as period_start,
            model_name,
            SUM(total_input_tokens)::BIGINT as input_tokens,
            SUM(total_output_tokens)::BIGINT as output_tokens,
            SUM(inference_count)::BIGINT as count
        FROM tensorzero.model_provider_statistics
        WHERE minute >= (
            SELECT COALESCE(MAX(date_trunc('month', minute)), '1970-01-01'::TIMESTAMPTZ)
            FROM tensorzero.model_provider_statistics
        ) - INTERVAL '12 months'
        GROUP BY period_start, model_name
        ORDER BY period_start DESC, model_name",
        );
    }

    // =========================================================================
    // Model latency quantiles query tests
    // =========================================================================

    #[test]
    fn test_build_model_latency_quantiles_raw_query_minute() {
        let qb = build_model_latency_quantiles_raw_query(&TimeWindow::Minute);
        assert_query_equals(
            qb.sql().as_str(),
            &format!(
                r"SELECT
            model_name,
            COALESCE(
                percentile_cont(ARRAY[{quantiles}]) WITHIN GROUP (ORDER BY response_time_ms),
                array_fill(NULL::double precision, ARRAY[{num_quantiles}])
            ) AS response_time_ms_quantiles,
            COALESCE(
                percentile_cont(ARRAY[{quantiles}]) WITHIN GROUP (ORDER BY ttft_ms),
                array_fill(NULL::double precision, ARRAY[{num_quantiles}])
            ) AS ttft_ms_quantiles,
            COUNT(*)::BIGINT as count
        FROM tensorzero.model_inferences
        WHERE created_at >= NOW() - INTERVAL '1 minute'
        GROUP BY model_name
        ORDER BY model_name",
                quantiles = *POSTGRES_QUANTILES_ARRAY_STRING,
                num_quantiles = POSTGRES_QUANTILES.len(),
            ),
        );
    }

    #[test]
    fn test_build_model_latency_quantiles_histogram_query_hour() {
        let qb = build_model_latency_quantiles_histogram_query(
            "tensorzero.model_latency_histogram_minute",
            "minute",
            Some("1 hour"),
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_provider_statistics WHERE TRUE
            AND minute >= NOW() - INTERVAL '1 hour'
            AND minute <= NOW()
            GROUP BY model_name",
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_latency_histogram_minute WHERE TRUE
            AND minute >= NOW() - INTERVAL '1 hour'
            AND minute <= NOW()
            GROUP BY model_name, metric, bucket_id",
        );
    }

    #[test]
    fn test_build_model_latency_quantiles_histogram_query_day() {
        let qb = build_model_latency_quantiles_histogram_query(
            "tensorzero.model_latency_histogram_hour",
            "hour",
            Some("1 day"),
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_provider_statistics WHERE TRUE
            AND minute >= NOW() - INTERVAL '1 day'
            AND minute <= NOW()
            GROUP BY model_name",
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_latency_histogram_hour WHERE TRUE
            AND hour >= NOW() - INTERVAL '1 day'
            AND hour <= NOW()
            GROUP BY model_name, metric, bucket_id",
        );
    }

    #[test]
    fn test_build_model_latency_quantiles_histogram_query_week() {
        let qb = build_model_latency_quantiles_histogram_query(
            "tensorzero.model_latency_histogram_hour",
            "hour",
            Some("1 week"),
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_provider_statistics WHERE TRUE
            AND minute >= NOW() - INTERVAL '1 week'
            AND minute <= NOW()
            GROUP BY model_name",
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_latency_histogram_hour WHERE TRUE
            AND hour >= NOW() - INTERVAL '1 week'
            AND hour <= NOW()
            GROUP BY model_name, metric, bucket_id",
        );
    }

    #[test]
    fn test_build_model_latency_quantiles_histogram_query_month() {
        let qb = build_model_latency_quantiles_histogram_query(
            "tensorzero.model_latency_histogram_hour",
            "hour",
            Some("1 month"),
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_provider_statistics WHERE TRUE
            AND minute >= NOW() - INTERVAL '1 month'
            AND minute <= NOW()
            GROUP BY model_name",
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_latency_histogram_hour WHERE TRUE
            AND hour >= NOW() - INTERVAL '1 month'
            AND hour <= NOW()
            GROUP BY model_name, metric, bucket_id",
        );
    }

    #[test]
    fn test_build_model_latency_quantiles_histogram_query_cumulative() {
        let qb = build_model_latency_quantiles_histogram_query(
            "tensorzero.model_latency_histogram_hour",
            "hour",
            None,
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_provider_statistics WHERE TRUE GROUP BY model_name",
        );
        assert_query_contains(
            qb.sql().as_str(),
            "FROM tensorzero.model_latency_histogram_hour WHERE TRUE GROUP BY model_name, metric, bucket_id",
        );
        assert_query_does_not_contain(qb.sql().as_str(), "NOW() - INTERVAL");
    }

    #[test]
    fn test_compute_model_latency_quantiles_from_histogram_buckets_empty_metrics() {
        let rows = vec![ModelLatencyHistogramBucketRow {
            model_name: "model_without_latency".to_string(),
            count: 12,
            metric: None,
            bucket_id: None,
            bucket_count: None,
        }];

        let datapoints = compute_model_latency_quantiles_from_histogram_buckets(rows);
        assert_eq!(
            datapoints.len(),
            1,
            "Expected one datapoint for one model count row"
        );
        assert_eq!(
            datapoints[0].count, 12,
            "Expected model count to match the aggregated model_provider_statistics count"
        );
        assert!(
            datapoints[0]
                .response_time_ms_quantiles
                .iter()
                .all(Option::is_none),
            "Expected response time quantiles to be all None when no histogram buckets are present"
        );
        assert!(
            datapoints[0].ttft_ms_quantiles.iter().all(Option::is_none),
            "Expected TTFT quantiles to be all None when no histogram buckets are present"
        );
    }

    #[test]
    fn test_compute_model_latency_quantiles_from_histogram_buckets_uses_geometric_mean() {
        let rows = vec![
            ModelLatencyHistogramBucketRow {
                model_name: "model_a".to_string(),
                count: 20,
                metric: Some(RESPONSE_TIME_MS_METRIC.to_string()),
                bucket_id: Some(0),
                bucket_count: Some(10.0),
            },
            ModelLatencyHistogramBucketRow {
                model_name: "model_a".to_string(),
                count: 20,
                metric: Some(RESPONSE_TIME_MS_METRIC.to_string()),
                bucket_id: Some(64),
                bucket_count: Some(10.0),
            },
            ModelLatencyHistogramBucketRow {
                model_name: "model_a".to_string(),
                count: 20,
                metric: Some(TTFT_MS_METRIC.to_string()),
                bucket_id: Some(-1),
                bucket_count: Some(20.0),
            },
        ];

        let datapoints = compute_model_latency_quantiles_from_histogram_buckets(rows);
        let p50_idx = POSTGRES_QUANTILES
            .iter()
            .position(|quantile| (*quantile - 0.5).abs() < f64::EPSILON)
            .expect("Expected quantiles to include P50");

        let response_p50 = datapoints[0].response_time_ms_quantiles[p50_idx]
            .expect("Expected response P50 to be present for non-empty histogram");
        let expected_response_p50 =
            (latency_bucket_lower_ms(64) * latency_bucket_upper_ms(64)).sqrt() as f32;
        assert!(
            (response_p50 - expected_response_p50).abs() < f32::EPSILON,
            "Expected response P50 to equal the geometric mean of the selected response bucket"
        );

        let ttft_p50 = datapoints[0].ttft_ms_quantiles[p50_idx]
            .expect("Expected TTFT P50 to be present for non-empty histogram");
        assert!(
            ttft_p50.abs() < f32::EPSILON,
            "Expected TTFT P50 to be zero when the selected bucket has lower bound 0"
        );
    }

    #[test]
    fn test_compute_model_latency_quantiles_from_histogram_buckets_empty_buckets() {
        let datapoints = compute_model_latency_quantiles_from_histogram_buckets(vec![]);
        assert!(
            datapoints.is_empty(),
            "Expected no datapoints when there are no histogram rows"
        );

        let zero_weight_rows = vec![ModelLatencyHistogramBucketRow {
            model_name: "model_zero_weight".to_string(),
            count: 3,
            metric: Some(RESPONSE_TIME_MS_METRIC.to_string()),
            bucket_id: Some(0),
            bucket_count: Some(0.0),
        }];

        let datapoints = compute_model_latency_quantiles_from_histogram_buckets(zero_weight_rows);
        assert_eq!(
            datapoints.len(),
            1,
            "Expected one datapoint for one model even when bucket weights are zero"
        );
        assert_eq!(
            datapoints[0].count, 3,
            "Expected model count to come from aggregated model_provider_statistics"
        );
        assert!(
            datapoints[0]
                .response_time_ms_quantiles
                .iter()
                .all(Option::is_none),
            "Expected response quantiles to be all None when histogram sample_count is zero"
        );
    }

    #[test]
    fn test_compute_model_latency_quantiles_from_histogram_buckets_very_few_elements() {
        let rows = vec![
            ModelLatencyHistogramBucketRow {
                model_name: "model_sparse".to_string(),
                count: 1,
                metric: Some(RESPONSE_TIME_MS_METRIC.to_string()),
                bucket_id: Some(0),
                bucket_count: Some(1.0),
            },
            ModelLatencyHistogramBucketRow {
                model_name: "model_sparse".to_string(),
                count: 1,
                metric: Some(TTFT_MS_METRIC.to_string()),
                bucket_id: Some(-1),
                bucket_count: Some(1.0),
            },
        ];

        let datapoints = compute_model_latency_quantiles_from_histogram_buckets(rows);
        assert_eq!(
            datapoints.len(),
            1,
            "Expected one datapoint for one sparse model"
        );
        assert_eq!(
            datapoints[0].count, 1,
            "Expected sparse model count to be preserved"
        );

        let expected_response =
            (latency_bucket_lower_ms(0) * latency_bucket_upper_ms(0)).sqrt() as f32;
        for quantile in &datapoints[0].response_time_ms_quantiles {
            let value = quantile.expect(
                "Expected every response quantile to be present when one response bucket exists",
            );
            assert!(
                (value - expected_response).abs() < f32::EPSILON,
                "Expected response quantiles to equal the geometric mean of the single response bucket"
            );
        }

        let expected_ttft = 0.0_f32;
        for quantile in &datapoints[0].ttft_ms_quantiles {
            let value = quantile
                .expect("Expected every TTFT quantile to be present when one TTFT bucket exists");
            assert!(
                (value - expected_ttft).abs() < f32::EPSILON,
                "Expected TTFT quantiles to be zero when the single TTFT bucket has lower bound 0"
            );
        }
    }

    // =========================================================================
    // Model inference query tests (existing)
    // =========================================================================

    #[test]
    fn test_build_get_model_inferences_query() {
        let inference_id = Uuid::nil();
        let qb = build_get_model_inferences_query(inference_id);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                i.id,
                i.inference_id,
                io.raw_request,
                io.raw_response,
                io.system,
                io.input_messages,
                io.output,
                i.input_tokens,
                i.output_tokens,
                i.response_time_ms,
                i.model_name,
                i.model_provider_name,
                i.ttft_ms,
                i.cached,
                i.finish_reason,
                i.snapshot_hash,
                i.created_at
            FROM tensorzero.model_inferences i
            LEFT JOIN tensorzero.model_inference_data io ON io.id = i.id AND io.created_at = i.created_at
            WHERE i.inference_id = $1
            ",
        );
    }

    #[test]
    fn test_build_insert_model_inferences_query_single_row() {
        let rows = vec![StoredModelInference {
            id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            raw_request: Some("request".to_string()),
            raw_response: Some("response".to_string()),
            system: Some("system".to_string()),
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(10),
            output_tokens: Some(20),
            response_time_ms: Some(100),
            model_name: "test_model".to_string(),
            model_provider_name: "test_provider".to_string(),
            ttft_ms: Some(50),
            cached: false,
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            timestamp: None,
        }];

        let qb = build_insert_model_inferences_query(&rows).expect("Should build metadata query");
        assert_query_equals(
            qb.sql().as_str(),
            r"
            INSERT INTO tensorzero.model_inferences (
                id, inference_id, input_tokens, output_tokens,
                response_time_ms, model_name, model_provider_name,
                ttft_ms, cached, finish_reason, snapshot_hash, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ",
        );

        let io_qb = build_insert_model_inference_data_query(&rows).expect("Should build IO query");
        assert_query_equals(
            io_qb.sql().as_str(),
            r"
            INSERT INTO tensorzero.model_inference_data (
                id, raw_request, raw_response, system,
                input_messages, output, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            ",
        );
    }

    #[test]
    fn test_build_insert_model_inferences_query_multiple_rows() {
        let rows = vec![
            StoredModelInference {
                id: Uuid::now_v7(),
                inference_id: Uuid::now_v7(),
                raw_request: Some("request1".to_string()),
                raw_response: Some("response1".to_string()),
                system: None,
                input_messages: Some(vec![]),
                output: Some(vec![]),
                input_tokens: None,
                output_tokens: None,
                response_time_ms: None,
                model_name: "model1".to_string(),
                model_provider_name: "provider1".to_string(),
                ttft_ms: None,
                cached: false,
                finish_reason: None,
                snapshot_hash: None,
                timestamp: None,
            },
            StoredModelInference {
                id: Uuid::now_v7(),
                inference_id: Uuid::now_v7(),
                raw_request: Some("request2".to_string()),
                raw_response: Some("response2".to_string()),
                system: Some("system2".to_string()),
                input_messages: Some(vec![]),
                output: Some(vec![]),
                input_tokens: Some(100),
                output_tokens: Some(200),
                response_time_ms: Some(500),
                model_name: "model2".to_string(),
                model_provider_name: "provider2".to_string(),
                ttft_ms: Some(25),
                cached: true,
                finish_reason: Some(FinishReason::ToolCall),
                snapshot_hash: None,
                timestamp: None,
            },
        ];

        let qb = build_insert_model_inferences_query(&rows).expect("Should build metadata query");
        assert_query_equals(
            qb.sql().as_str(),
            r"
            INSERT INTO tensorzero.model_inferences (
                id, inference_id, input_tokens, output_tokens,
                response_time_ms, model_name, model_provider_name,
                ttft_ms, cached, finish_reason, snapshot_hash, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12),
            ($13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
            ",
        );

        let io_qb = build_insert_model_inference_data_query(&rows).expect("Should build IO query");
        assert_query_equals(
            io_qb.sql().as_str(),
            r"
            INSERT INTO tensorzero.model_inference_data (
                id, raw_request, raw_response, system,
                input_messages, output, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7),
            ($8, $9, $10, $11, $12, $13, $14)
            ",
        );
    }
}
