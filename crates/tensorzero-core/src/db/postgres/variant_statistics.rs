//! Postgres queries for variant statistics.

use async_trait::async_trait;
use sqlx::QueryBuilder;

use super::PostgresConnectionInfo;
use crate::db::variant_statistics::{
    GetVariantStatisticsParams, VariantStatisticsQueries, VariantStatisticsRow,
};
use crate::error::Error;

#[derive(Debug, sqlx::FromRow)]
struct VariantStatisticsDbRow {
    function_name: String,
    variant_name: String,
    inference_count: i64,
    total_input_tokens: Option<i64>,
    total_output_tokens: Option<i64>,
    total_cost: Option<rust_decimal::Decimal>,
    count_with_cost: Option<i64>,
    total_provider_cache_read_input_tokens: Option<i64>,
    total_provider_cache_write_input_tokens: Option<i64>,
}

impl From<VariantStatisticsDbRow> for VariantStatisticsRow {
    fn from(row: VariantStatisticsDbRow) -> Self {
        Self {
            function_name: row.function_name,
            variant_name: row.variant_name,
            inference_count: row.inference_count as u64,
            total_input_tokens: row.total_input_tokens.map(|v| v as u64),
            total_output_tokens: row.total_output_tokens.map(|v| v as u64),
            total_cost: row.total_cost,
            count_with_cost: row.count_with_cost.map(|v| v as u64),
            total_provider_cache_read_input_tokens: row
                .total_provider_cache_read_input_tokens
                .map(|v| v as u64),
            total_provider_cache_write_input_tokens: row
                .total_provider_cache_write_input_tokens
                .map(|v| v as u64),
            // Postgres doesn't have latency quantiles in the rollup table
            processing_time_ms_quantiles: None,
            ttft_ms_quantiles: None,
        }
    }
}

#[async_trait]
impl VariantStatisticsQueries for PostgresConnectionInfo {
    async fn get_variant_statistics(
        &self,
        params: &GetVariantStatisticsParams,
    ) -> Result<Vec<VariantStatisticsRow>, Error> {
        let pool = self.get_pool_result().map_err(|e| e.log())?;

        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
            r"
            SELECT
                function_name,
                variant_name,
                SUM(inference_count)::BIGINT AS inference_count,
                SUM(total_input_tokens)::BIGINT AS total_input_tokens,
                SUM(total_output_tokens)::BIGINT AS total_output_tokens,
                SUM(total_cost) AS total_cost,
                SUM(count_with_cost)::BIGINT AS count_with_cost,
                SUM(total_provider_cache_read_input_tokens)::BIGINT AS total_provider_cache_read_input_tokens,
                SUM(total_provider_cache_write_input_tokens)::BIGINT AS total_provider_cache_write_input_tokens
            FROM tensorzero.variant_statistics
            WHERE function_name = ",
        );
        qb.push_bind(&params.function_name);

        if let Some(variant_names) = &params.variant_names
            && !variant_names.is_empty()
        {
            qb.push(" AND variant_name = ANY(");
            qb.push_bind(variant_names);
            qb.push(")");
        }

        if let Some(after) = &params.after {
            qb.push(" AND minute >= ");
            qb.push_bind(after);
        }

        qb.push(" GROUP BY function_name, variant_name ORDER BY function_name, variant_name");

        let rows: Vec<VariantStatisticsDbRow> = qb.build_query_as().fetch_all(pool).await?;

        Ok(rows.into_iter().map(Into::into).collect())
    }

    fn get_variant_statistics_quantiles(&self) -> Option<&[f64]> {
        None
    }
}
