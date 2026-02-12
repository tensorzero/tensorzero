//! Model inference queries for Postgres.
//!
//! This module implements read and write operations for the model_inferences table in Postgres.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lazy_static::lazy_static;
use sqlx::types::Json;
use sqlx::{PgPool, QueryBuilder, Row};
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::db::model_inferences::ModelInferenceQueries;
use crate::db::query_helpers::uuid_to_datetime;
use crate::db::{ModelLatencyDatapoint, ModelUsageTimePoint, TimeWindow};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{
    ContentBlockOutput, FinishReason, StoredModelInference, StoredRequestMessage,
};

use super::PostgresConnectionInfo;

/// Quantiles used for Postgres latency queries.
pub const POSTGRES_QUANTILES: &[f64; 17] = &[
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995,
    0.999,
];
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

        let pool = self.get_pool_result()?;
        let mut qb = build_insert_model_inferences_query(rows)?;
        qb.build().execute(pool).await?;
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

        let mut query_builder = build_model_latency_quantiles_query(&time_window);
        let rows: Vec<ModelLatencyDatapoint> =
            query_builder.build_query_as().fetch_all(pool).await?;

        Ok(rows)
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
            id,
            inference_id,
            raw_request,
            raw_response,
            system,
            input_messages,
            output,
            input_tokens,
            output_tokens,
            response_time_ms,
            model_name,
            model_provider_name,
            ttft_ms,
            cached,
            finish_reason,
            snapshot_hash,
            cost,
            created_at
        FROM tensorzero.model_inferences
        WHERE inference_id = ",
    );
    qb.push_bind(inference_id);

    qb
}

/// Builds a query to insert model inferences.
fn build_insert_model_inferences_query(
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
            id, inference_id, raw_request, raw_response, system,
            input_messages, output, input_tokens, output_tokens,
            response_time_ms, model_name, model_provider_name,
            ttft_ms, cached, finish_reason, snapshot_hash, cost, created_at
        ) ",
    );

    qb.push_values(rows.iter().zip(&timestamps), |mut b, (row, created_at)| {
        let snapshot_hash_bytes: Option<Vec<u8>> =
            row.snapshot_hash.as_ref().map(|h| h.as_bytes().to_vec());

        b.push_bind(row.id)
            .push_bind(row.inference_id)
            .push_bind(&row.raw_request)
            .push_bind(&row.raw_response)
            .push_bind(&row.system)
            .push_bind(Json(&row.input_messages))
            .push_bind(Json(&row.output))
            .push_bind(row.input_tokens.map(|v| v as i32))
            .push_bind(row.output_tokens.map(|v| v as i32))
            .push_bind(row.response_time_ms.map(|v| v as i32))
            .push_bind(&row.model_name)
            .push_bind(&row.model_provider_name)
            .push_bind(row.ttft_ms.map(|v| v as i32))
            .push_bind(row.cached)
            .push_bind(row.finish_reason)
            .push_bind(snapshot_hash_bytes)
            .push_bind(row.cost)
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

/// Builds the query for model latency quantiles.
/// For Minute, queries raw data. For Hour/Day/Week/Month/Cumulative, queries materialized views.
fn build_model_latency_quantiles_query(time_window: &TimeWindow) -> QueryBuilder<sqlx::Postgres> {
    // For Hour/Day/Week/Month/Cumulative, use precomputed materialized views
    // For Minute, compute from raw data since the data volume is small
    match time_window {
        TimeWindow::Minute => build_model_latency_quantiles_raw_query(time_window),
        TimeWindow::Hour => {
            build_model_latency_quantiles_view_query("tensorzero.model_latency_quantiles_hour")
        }
        TimeWindow::Day => {
            build_model_latency_quantiles_view_query("tensorzero.model_latency_quantiles_day")
        }
        TimeWindow::Week => {
            build_model_latency_quantiles_view_query("tensorzero.model_latency_quantiles_week")
        }
        TimeWindow::Month => {
            build_model_latency_quantiles_view_query("tensorzero.model_latency_quantiles_month")
        }
        TimeWindow::Cumulative => {
            build_model_latency_quantiles_view_query("tensorzero.model_latency_quantiles")
        }
    }
}

/// Builds a query to read from a precomputed materialized view.
fn build_model_latency_quantiles_view_query(table_name: &str) -> QueryBuilder<sqlx::Postgres> {
    let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        "SELECT
            model_name,
            response_time_ms_quantiles,
            ttft_ms_quantiles,
            count
        FROM ",
    );
    qb.push(table_name);
    qb.push(" ORDER BY model_name");
    qb
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
        let raw_request: String = row.try_get("raw_request")?;
        let raw_response: String = row.try_get("raw_response")?;
        let system: Option<String> = row.try_get("system")?;
        let input_messages: Json<Vec<StoredRequestMessage>> = row.try_get("input_messages")?;
        let output: Json<Vec<ContentBlockOutput>> = row.try_get("output")?;
        let input_tokens: Option<i32> = row.try_get("input_tokens")?;
        let output_tokens: Option<i32> = row.try_get("output_tokens")?;
        let response_time_ms: Option<i32> = row.try_get("response_time_ms")?;
        let model_name: String = row.try_get("model_name")?;
        let model_provider_name: String = row.try_get("model_provider_name")?;
        let ttft_ms: Option<i32> = row.try_get("ttft_ms")?;
        let cached: bool = row.try_get("cached")?;
        let finish_reason: Option<FinishReason> = row.try_get("finish_reason")?;
        let snapshot_hash_bytes: Option<Vec<u8>> = row.try_get("snapshot_hash")?;
        let cost: Option<rust_decimal::Decimal> = row.try_get("cost")?;
        let created_at: DateTime<Utc> = row.try_get("created_at")?;

        // Convert snapshot_hash from bytes
        let snapshot_hash = snapshot_hash_bytes.map(|bytes| SnapshotHash::from_bytes(&bytes));

        Ok(StoredModelInference {
            id,
            inference_id,
            raw_request,
            raw_response,
            system,
            input_messages: input_messages.0,
            output: output.0,
            input_tokens: input_tokens.map(|v| v as u32),
            output_tokens: output_tokens.map(|v| v as u32),
            response_time_ms: response_time_ms.map(|v| v as u32),
            model_name,
            model_provider_name,
            ttft_ms: ttft_ms.map(|v| v as u32),
            cached,
            finish_reason,
            snapshot_hash,
            cost,
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
    use crate::db::test_helpers::assert_query_equals;

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
    fn test_build_model_latency_quantiles_query_minute() {
        let qb = build_model_latency_quantiles_query(&TimeWindow::Minute);
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
    fn test_build_model_latency_quantiles_query_hour() {
        let qb = build_model_latency_quantiles_query(&TimeWindow::Hour);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT
            model_name,
            response_time_ms_quantiles,
            ttft_ms_quantiles,
            count
        FROM tensorzero.model_latency_quantiles_hour ORDER BY model_name",
        );
    }

    #[test]
    fn test_build_model_latency_quantiles_query_day() {
        let qb = build_model_latency_quantiles_query(&TimeWindow::Day);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT
            model_name,
            response_time_ms_quantiles,
            ttft_ms_quantiles,
            count
        FROM tensorzero.model_latency_quantiles_day ORDER BY model_name",
        );
    }

    #[test]
    fn test_build_model_latency_quantiles_query_week() {
        let qb = build_model_latency_quantiles_query(&TimeWindow::Week);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT
            model_name,
            response_time_ms_quantiles,
            ttft_ms_quantiles,
            count
        FROM tensorzero.model_latency_quantiles_week ORDER BY model_name",
        );
    }

    #[test]
    fn test_build_model_latency_quantiles_query_month() {
        let qb = build_model_latency_quantiles_query(&TimeWindow::Month);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT
            model_name,
            response_time_ms_quantiles,
            ttft_ms_quantiles,
            count
        FROM tensorzero.model_latency_quantiles_month ORDER BY model_name",
        );
    }

    #[test]
    fn test_build_model_latency_quantiles_query_cumulative() {
        let qb = build_model_latency_quantiles_query(&TimeWindow::Cumulative);
        assert_query_equals(
            qb.sql().as_str(),
            r"SELECT
            model_name,
            response_time_ms_quantiles,
            ttft_ms_quantiles,
            count
        FROM tensorzero.model_latency_quantiles ORDER BY model_name",
        );
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
                id,
                inference_id,
                raw_request,
                raw_response,
                system,
                input_messages,
                output,
                input_tokens,
                output_tokens,
                response_time_ms,
                model_name,
                model_provider_name,
                ttft_ms,
                cached,
                finish_reason,
                snapshot_hash,
                cost,
                created_at
            FROM tensorzero.model_inferences
            WHERE inference_id = $1
            ",
        );
    }

    /// Number of columns in the `model_inferences` INSERT statement.
    /// Update this constant when adding/removing columns in `build_insert_model_inferences_query`.
    const MODEL_INFERENCES_INSERT_COLUMNS: usize = 18;

    /// Generate the expected VALUES clause for a multi-row INSERT with `num_rows` rows
    /// and `cols_per_row` bind parameters per row.
    ///
    /// Example: `expected_values_clause(2, 3)` => `"($1, $2, $3), ($4, $5, $6)"`
    fn expected_values_clause(num_rows: usize, cols_per_row: usize) -> String {
        (0..num_rows)
            .map(|row| {
                let params: Vec<String> = (1..=cols_per_row)
                    .map(|col| format!("${}", row * cols_per_row + col))
                    .collect();
                format!("({})", params.join(", "))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    #[test]
    fn test_build_insert_model_inferences_query_single_row() {
        let rows = vec![StoredModelInference {
            id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            raw_request: "request".to_string(),
            raw_response: "response".to_string(),
            system: Some("system".to_string()),
            input_messages: vec![],
            output: vec![],
            input_tokens: Some(10),
            output_tokens: Some(20),
            response_time_ms: Some(100),
            model_name: "test_model".to_string(),
            model_provider_name: "test_provider".to_string(),
            ttft_ms: Some(50),
            cached: false,
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            cost: None,
            timestamp: None,
        }];

        let qb = build_insert_model_inferences_query(&rows).expect("Should build query");
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        let expected = format!(
            r"
            INSERT INTO tensorzero.model_inferences (
                id, inference_id, raw_request, raw_response, system,
                input_messages, output, input_tokens, output_tokens,
                response_time_ms, model_name, model_provider_name,
                ttft_ms, cached, finish_reason, snapshot_hash, cost, created_at
            ) VALUES {}
            ",
            expected_values_clause(1, MODEL_INFERENCES_INSERT_COLUMNS)
        );

        assert_query_equals(sql, &expected);
    }

    #[test]
    fn test_build_insert_model_inferences_query_multiple_rows() {
        let rows = vec![
            StoredModelInference {
                id: Uuid::now_v7(),
                inference_id: Uuid::now_v7(),
                raw_request: "request1".to_string(),
                raw_response: "response1".to_string(),
                system: None,
                input_messages: vec![],
                output: vec![],
                input_tokens: None,
                output_tokens: None,
                response_time_ms: None,
                model_name: "model1".to_string(),
                model_provider_name: "provider1".to_string(),
                ttft_ms: None,
                cached: false,
                finish_reason: None,
                snapshot_hash: None,
                cost: None,
                timestamp: None,
            },
            StoredModelInference {
                id: Uuid::now_v7(),
                inference_id: Uuid::now_v7(),
                raw_request: "request2".to_string(),
                raw_response: "response2".to_string(),
                system: Some("system2".to_string()),
                input_messages: vec![],
                output: vec![],
                input_tokens: Some(100),
                output_tokens: Some(200),
                response_time_ms: Some(500),
                model_name: "model2".to_string(),
                model_provider_name: "provider2".to_string(),
                ttft_ms: Some(25),
                cached: true,
                finish_reason: Some(FinishReason::ToolCall),
                snapshot_hash: None,
                cost: None,
                timestamp: None,
            },
        ];

        let qb = build_insert_model_inferences_query(&rows).expect("Should build query");
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        let expected = format!(
            r"
            INSERT INTO tensorzero.model_inferences (
                id, inference_id, raw_request, raw_response, system,
                input_messages, output, input_tokens, output_tokens,
                response_time_ms, model_name, model_provider_name,
                ttft_ms, cached, finish_reason, snapshot_hash, cost, created_at
            ) VALUES {}
            ",
            expected_values_clause(2, MODEL_INFERENCES_INSERT_COLUMNS)
        );

        assert_query_equals(sql, &expected);
    }
}
