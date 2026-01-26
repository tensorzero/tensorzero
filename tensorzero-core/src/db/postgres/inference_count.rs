//! Postgres queries for inference count.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{PgPool, QueryBuilder, Row};

use super::PostgresConnectionInfo;
use crate::db::TimeWindow;
use crate::db::inference_count::{
    CountByVariant, CountInferencesParams, CountInferencesWithDemonstrationFeedbacksParams,
    CountInferencesWithFeedbackParams, FunctionInferenceCount,
    GetFunctionThroughputByVariantParams, InferenceCountQueries, VariantThroughput,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfigType;

/// Builds and executes a count query for inferences.
async fn count_inferences(
    pool: &PgPool,
    function_type: FunctionConfigType,
    function_name: &str,
    variant_name: Option<&str>,
) -> Result<i64, sqlx::Error> {
    let table = function_type.postgres_table_name();

    let mut qb = QueryBuilder::new("SELECT COUNT(*) FROM ");
    qb.push(table);
    qb.push(" WHERE function_name = ");
    qb.push_bind(function_name);

    if let Some(variant) = variant_name {
        qb.push(" AND variant_name = ");
        qb.push_bind(variant);
    }

    qb.build_query_scalar().fetch_one(pool).await
}

/// Builds and executes a count-by-variant query for inferences.
async fn count_by_variant(
    pool: &PgPool,
    function_type: FunctionConfigType,
    function_name: &str,
    variant_name: Option<&str>,
) -> Result<Vec<CountByVariant>, sqlx::Error> {
    let table = function_type.postgres_table_name();

    let mut qb = QueryBuilder::new(
        r#"SELECT
            variant_name,
            COUNT(*) AS inference_count,
            to_char(MAX(created_at), 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"') AS last_used_at
        FROM "#,
    );
    qb.push(table);
    qb.push(" WHERE function_name = ");
    qb.push_bind(function_name);

    if let Some(variant) = variant_name {
        qb.push(" AND variant_name = ");
        qb.push_bind(variant);
    }

    qb.push(" GROUP BY variant_name ORDER BY inference_count DESC");

    let rows = qb.build().fetch_all(pool).await?;

    Ok(rows
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
        .collect())
}

/// Builds and executes a throughput-by-variant query.
async fn throughput_by_variant(
    pool: &PgPool,
    function_name: &str,
    time_window: TimeWindow,
    max_periods: u32,
) -> Result<Vec<VariantThroughput>, Error> {
    let rows = if time_window == TimeWindow::Cumulative {
        // For cumulative, return all-time data grouped by variant with fixed epoch start
        let mut qb = QueryBuilder::new(
            r"SELECT
                '1970-01-01T00:00:00.000Z'::text AS period_start,
                variant_name,
                COUNT(*)::INT AS count
            FROM (
                SELECT variant_name FROM tensorzero.chat_inferences WHERE function_name = ",
        );
        qb.push_bind(function_name);
        qb.push(
            " UNION ALL SELECT variant_name FROM tensorzero.json_inferences WHERE function_name = ",
        );
        qb.push_bind(function_name);
        qb.push(") AS combined GROUP BY variant_name ORDER BY variant_name DESC");

        qb.build().fetch_all(pool).await?
    } else {
        let unit = time_window.to_postgres_time_unit();

        let mut qb = QueryBuilder::new(
            "WITH combined AS (
                SELECT variant_name, created_at FROM tensorzero.chat_inferences WHERE function_name = ",
        );
        qb.push_bind(function_name);
        qb.push(" UNION ALL SELECT variant_name, created_at FROM tensorzero.json_inferences WHERE function_name = ");
        qb.push_bind(function_name);
        qb.push(
            "),
            max_time AS (
                SELECT MAX(created_at) AS max_ts FROM combined
            )
            SELECT
                to_char(date_trunc('",
        );
        qb.push(unit);
        qb.push(
            r#"', c.created_at), 'YYYY-MM-DD"T"HH24:MI:SS.000"Z"') AS period_start,
                c.variant_name,
                COUNT(*)::INT AS count
            FROM combined c, max_time m
            WHERE c.created_at >= m.max_ts - INTERVAL '1 "#,
        );
        qb.push(unit);
        qb.push("' * (");
        qb.push_bind(max_periods as i32);
        qb.push(" + 1) GROUP BY date_trunc('");
        qb.push(unit);
        qb.push("', c.created_at), c.variant_name ORDER BY period_start DESC, variant_name DESC");

        qb.build().fetch_all(pool).await?
    };

    let variant_throughputs = rows
        .into_iter()
        .map(|row| {
            let period_start_str: String = row.get("period_start");
            let period_start = DateTime::parse_from_rfc3339(&period_start_str)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|err| {
                    Error::new(ErrorDetails::PostgresResult {
                        result_type: "variant_throughput",
                        message: format!(
                            "Failed to parse `period_start` value `{period_start_str}`: {err}"
                        ),
                    })
                })?;
            let variant_name: String = row.get("variant_name");
            let count: i32 = row.get("count");
            Ok(VariantThroughput {
                period_start,
                variant_name,
                count: count as u32,
            })
        })
        .collect::<Result<Vec<VariantThroughput>, Error>>()?;
    Ok(variant_throughputs)
}

#[async_trait]
impl InferenceCountQueries for PostgresConnectionInfo {
    async fn count_inferences_for_function(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;
        let count = count_inferences(
            pool,
            params.function_type,
            params.function_name,
            params.variant_name,
        )
        .await
        .map_err(Error::from)?;
        Ok(count as u64)
    }

    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error> {
        let pool = self.get_pool_result()?;
        count_by_variant(
            pool,
            params.function_type,
            params.function_name,
            params.variant_name,
        )
        .await
        .map_err(Error::from)
    }

    async fn count_inferences_with_feedback(
        &self,
        _params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error> {
        // TODO(#5691): Implement when feedback tables are added in step-2
        Err(Error::new(ErrorDetails::NotImplemented {
            message: "count_inferences_with_feedback not yet implemented for Postgres".to_string(),
        }))
    }

    async fn count_inferences_with_demonstration_feedback(
        &self,
        _params: CountInferencesWithDemonstrationFeedbacksParams<'_>,
    ) -> Result<u64, Error> {
        // TODO(#5691): Implement when feedback tables are added in step-2
        Err(Error::new(ErrorDetails::NotImplemented {
            message:
                "count_inferences_with_demonstration_feedback not yet implemented for Postgres"
                    .to_string(),
        }))
    }

    async fn count_inferences_for_episode(&self, episode_id: uuid::Uuid) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;

        let mut qb = QueryBuilder::new(
            r"SELECT COUNT(*) FROM (
                SELECT id FROM tensorzero.chat_inferences WHERE episode_id = ",
        );
        qb.push_bind(episode_id);
        qb.push(" UNION ALL SELECT id FROM tensorzero.json_inferences WHERE episode_id = ");
        qb.push_bind(episode_id);
        qb.push(") AS combined");

        let count: i64 = qb
            .build_query_scalar()
            .fetch_one(pool)
            .await
            .map_err(Error::from)?;

        Ok(count as u64)
    }

    async fn get_function_throughput_by_variant(
        &self,
        params: GetFunctionThroughputByVariantParams<'_>,
    ) -> Result<Vec<VariantThroughput>, Error> {
        let pool = self.get_pool_result()?;
        throughput_by_variant(
            pool,
            params.function_name,
            params.time_window,
            params.max_periods,
        )
        .await
    }

    async fn list_functions_with_inference_count(
        &self,
    ) -> Result<Vec<FunctionInferenceCount>, Error> {
        let pool = self.get_pool_result()?;

        let rows = sqlx::query(
            r"
            SELECT
                function_name,
                MAX(created_at) AS last_inference_timestamp,
                COUNT(*)::INT AS inference_count
            FROM (
                SELECT function_name, created_at FROM tensorzero.chat_inferences
                UNION ALL
                SELECT function_name, created_at FROM tensorzero.json_inferences
            ) AS combined
            GROUP BY function_name
            ORDER BY last_inference_timestamp DESC
            ",
        )
        .fetch_all(pool)
        .await
        .map_err(Error::from)?;

        let results = rows
            .into_iter()
            .map(|row| {
                let function_name: String = row.get("function_name");
                let last_inference_timestamp: DateTime<Utc> = row.get("last_inference_timestamp");
                let inference_count: i32 = row.get("inference_count");
                FunctionInferenceCount {
                    function_name,
                    last_inference_timestamp,
                    inference_count: inference_count as u32,
                }
            })
            .collect();

        Ok(results)
    }
}
