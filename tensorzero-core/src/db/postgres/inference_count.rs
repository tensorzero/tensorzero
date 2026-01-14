//! PostgreSQL queries for inference count.

use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use sqlx::{AssertSqlSafe, Row};

use crate::db::TimeWindow;
use crate::db::inference_count::{
    CountByVariant, CountInferencesParams, CountInferencesWithDemonstrationFeedbacksParams,
    CountInferencesWithFeedbackParams, FunctionInferenceCount,
    GetFunctionThroughputByVariantParams, InferenceCountQueries, VariantThroughput,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfigType;

use super::PostgresConnectionInfo;

/// Maps FunctionConfigType to PostgreSQL table name.
/// PostgreSQL uses lowercase snake_case table names in the tensorzero schema.
fn pg_table_name(function_type: FunctionConfigType) -> &'static str {
    match function_type {
        FunctionConfigType::Chat => "tensorzero.chat_inference",
        FunctionConfigType::Json => "tensorzero.json_inference",
    }
}

#[async_trait]
impl InferenceCountQueries for PostgresConnectionInfo {
    async fn count_inferences_for_function(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;
        let table_name = pg_table_name(params.function_type);

        let count: i64 = match params.variant_name {
            Some(variant_name) => {
                let query = format!(
                    "SELECT COUNT(*) AS count FROM {table_name} WHERE function_name = $1 AND variant_name = $2"
                );
                sqlx::query_scalar(AssertSqlSafe(query))
                    .bind(params.function_name)
                    .bind(variant_name)
                    .fetch_one(pool)
                    .await?
            }
            None => {
                let query =
                    format!("SELECT COUNT(*) AS count FROM {table_name} WHERE function_name = $1");
                sqlx::query_scalar(AssertSqlSafe(query))
                    .bind(params.function_name)
                    .fetch_one(pool)
                    .await?
            }
        };

        Ok(count as u64)
    }

    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error> {
        let pool = self.get_pool_result()?;
        let table_name = pg_table_name(params.function_type);

        let rows = match params.variant_name {
            Some(variant_name) => {
                let query = format!(
                    r#"SELECT
                        variant_name,
                        COUNT(*) AS inference_count,
                        TO_CHAR(MAX(timestamp) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"') AS last_used_at
                    FROM {table_name}
                    WHERE function_name = $1 AND variant_name = $2
                    GROUP BY variant_name
                    ORDER BY inference_count DESC"#
                );
                sqlx::query(AssertSqlSafe(query))
                    .bind(params.function_name)
                    .bind(variant_name)
                    .fetch_all(pool)
                    .await?
            }
            None => {
                let query = format!(
                    r#"SELECT
                        variant_name,
                        COUNT(*) AS inference_count,
                        TO_CHAR(MAX(timestamp) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"') AS last_used_at
                    FROM {table_name}
                    WHERE function_name = $1
                    GROUP BY variant_name
                    ORDER BY inference_count DESC"#
                );
                sqlx::query(AssertSqlSafe(query))
                    .bind(params.function_name)
                    .fetch_all(pool)
                    .await?
            }
        };

        let result = rows
            .into_iter()
            .map(|row| {
                let variant_name: String = row.get("variant_name");
                let inference_count: i64 = row.get("inference_count");
                let last_used_at: String = row.get("last_used_at");

                CountByVariant {
                    variant_name,
                    inference_count: inference_count as u64,
                    last_used_at,
                }
            })
            .collect();

        Ok(result)
    }

    async fn count_inferences_with_feedback(
        &self,
        _params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error> {
        Err(Error::new(ErrorDetails::NotImplemented {
            message: "count_inferences_with_feedback requires feedback tables which are not yet implemented for PostgreSQL".to_string(),
        }))
    }

    async fn count_inferences_with_demonstration_feedback(
        &self,
        _params: CountInferencesWithDemonstrationFeedbacksParams<'_>,
    ) -> Result<u64, Error> {
        Err(Error::new(ErrorDetails::NotImplemented {
            message: "count_inferences_with_demonstration_feedback requires DemonstrationFeedback table which is not yet implemented for PostgreSQL".to_string(),
        }))
    }

    async fn count_inferences_for_episode(&self, episode_id: uuid::Uuid) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;

        // Count across both tables using UNION ALL
        let count: i64 = sqlx::query_scalar(
            r"SELECT COALESCE(SUM(count), 0)::bigint AS count FROM (
                SELECT COUNT(*) AS count FROM tensorzero.chat_inference WHERE episode_id = $1
                UNION ALL
                SELECT COUNT(*) AS count FROM tensorzero.json_inference WHERE episode_id = $1
            ) t",
        )
        .bind(episode_id)
        .fetch_one(pool)
        .await?;

        Ok(count as u64)
    }

    async fn get_function_throughput_by_variant(
        &self,
        params: GetFunctionThroughputByVariantParams<'_>,
    ) -> Result<Vec<VariantThroughput>, Error> {
        let pool = self.get_pool_result()?;

        let rows = match params.time_window {
            TimeWindow::Cumulative => {
                // For cumulative, return all-time data grouped by variant with fixed epoch start
                sqlx::query(
                    r"SELECT
                        '1970-01-01T00:00:00.000Z' AS period_start,
                        variant_name,
                        COUNT(*)::integer AS count
                    FROM (
                        SELECT variant_name FROM tensorzero.chat_inference WHERE function_name = $1
                        UNION ALL
                        SELECT variant_name FROM tensorzero.json_inference WHERE function_name = $1
                    ) t
                    GROUP BY variant_name
                    ORDER BY variant_name DESC",
                )
                .bind(params.function_name)
                .fetch_all(pool)
                .await?
            }
            _ => {
                // Calculate time delta
                let time_window_duration = params.time_window.to_duration();
                let time_delta = time_window_duration * (params.max_periods + 1);
                let interval_str = params.time_window.to_postgres_interval_string();

                // Build dynamic query with time filtering relative to max timestamp
                let query = format!(
                    r#"WITH combined AS (
                        SELECT function_name, variant_name, timestamp
                        FROM tensorzero.chat_inference
                        WHERE function_name = $1
                        UNION ALL
                        SELECT function_name, variant_name, timestamp
                        FROM tensorzero.json_inference
                        WHERE function_name = $1
                    ),
                    max_ts AS (
                        SELECT MAX(timestamp) AS max_timestamp FROM combined
                    )
                    SELECT
                        TO_CHAR(date_trunc('{interval_str}', timestamp) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"') AS period_start,
                        variant_name,
                        COUNT(*)::integer AS count
                    FROM combined, max_ts
                    WHERE timestamp >= max_ts.max_timestamp - INTERVAL '{} seconds'
                    GROUP BY period_start, variant_name
                    ORDER BY period_start DESC, variant_name DESC"#,
                    time_delta.as_secs()
                );

                sqlx::query(AssertSqlSafe(query))
                    .bind(params.function_name)
                    .fetch_all(pool)
                    .await?
            }
        };

        let result = rows
            .into_iter()
            .map(|row| {
                let period_start_str: String = row.get("period_start");
                let variant_name: String = row.get("variant_name");
                let count: i32 = row.get("count");

                // Parse the timestamp string to DateTime<Utc>
                // Format: YYYY-MM-DDTHH:MI:SS.MSSZ
                let period_start = if period_start_str == "1970-01-01T00:00:00.000Z" {
                    Utc.with_ymd_and_hms(1970, 1, 1, 0, 0, 0).unwrap()
                } else {
                    DateTime::parse_from_rfc3339(&period_start_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now())
                };

                VariantThroughput {
                    period_start,
                    variant_name,
                    count: count as u32,
                }
            })
            .collect();

        Ok(result)
    }

    async fn list_functions_with_inference_count(
        &self,
    ) -> Result<Vec<FunctionInferenceCount>, Error> {
        let pool = self.get_pool_result()?;

        let rows = sqlx::query(
            r#"SELECT
                function_name,
                TO_CHAR(MAX(timestamp) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"') AS last_inference_timestamp,
                COUNT(*)::integer AS inference_count
            FROM (
                SELECT function_name, timestamp FROM tensorzero.chat_inference
                UNION ALL
                SELECT function_name, timestamp FROM tensorzero.json_inference
            ) t
            GROUP BY function_name
            ORDER BY last_inference_timestamp DESC"#,
        )
        .fetch_all(pool)
        .await?;

        let result = rows
            .into_iter()
            .map(|row| {
                let function_name: String = row.get("function_name");
                let last_inference_timestamp_str: String = row.get("last_inference_timestamp");
                let inference_count: i32 = row.get("inference_count");

                // Parse the timestamp string to DateTime<Utc>
                let last_inference_timestamp =
                    DateTime::parse_from_rfc3339(&last_inference_timestamp_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());

                FunctionInferenceCount {
                    function_name,
                    last_inference_timestamp,
                    inference_count: inference_count as u32,
                }
            })
            .collect();

        Ok(result)
    }
}
