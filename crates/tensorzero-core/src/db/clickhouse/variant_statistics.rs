//! ClickHouse queries for variant statistics.

use std::collections::HashMap;

use async_trait::async_trait;

use super::ClickHouseConnectionInfo;
use super::escape_string_for_clickhouse_literal;
use super::migration_manager::migrations::migration_0037::{QUANTILES, quantiles_sql_args};
use crate::db::variant_statistics::{
    GetVariantStatisticsParams, VariantStatisticsQueries, VariantStatisticsRow,
};
use crate::error::{Error, ErrorDetails};

/// Format for ClickHouse `DateTime` columns (second precision, no sub-second digits).
const CLICKHOUSE_SECOND_PRECISION_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

#[async_trait]
impl VariantStatisticsQueries for ClickHouseConnectionInfo {
    async fn get_variant_statistics(
        &self,
        params: &GetVariantStatisticsParams,
    ) -> Result<Vec<VariantStatisticsRow>, Error> {
        let qs = quantiles_sql_args();
        let function_name_str = &params.function_name;

        let mut where_clauses = vec!["function_name = {function_name:String}".to_string()];
        let mut query_params: HashMap<&str, &str> =
            HashMap::from([("function_name", function_name_str.as_str())]);

        let variant_names_param;
        if let Some(variant_names) = &params.variant_names
            && !variant_names.is_empty()
        {
            variant_names_param = format!(
                "[{}]",
                variant_names
                    .iter()
                    .map(|v| format!("'{}'", escape_string_for_clickhouse_literal(v)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            query_params.insert("variant_names", variant_names_param.as_str());
            where_clauses.push("variant_name IN {variant_names:Array(String)}".to_string());
        }

        let after_str;
        if let Some(after) = &params.after {
            after_str = after.format(CLICKHOUSE_SECOND_PRECISION_FORMAT).to_string();
            query_params.insert("after", after_str.as_str());
            where_clauses.push("minute >= {after:DateTime}".to_string());
        }

        let before_str;
        if let Some(before) = &params.before {
            before_str = before
                .format(CLICKHOUSE_SECOND_PRECISION_FORMAT)
                .to_string();
            query_params.insert("before", before_str.as_str());
            where_clauses.push("minute < {before:DateTime}".to_string());
        }

        let where_clause = where_clauses.join(" AND ");

        let query = format!(
            r"
            SELECT
                function_name,
                variant_name,
                countMerge(count) as inference_count,
                sumMerge(total_input_tokens) as total_input_tokens,
                sumMerge(total_output_tokens) as total_output_tokens,
                sumMerge(total_cost) as total_cost,
                countMerge(count_with_cost) as count_with_cost,
                sumMerge(total_provider_cache_read_input_tokens) as total_provider_cache_read_input_tokens,
                sumMerge(total_provider_cache_write_input_tokens) as total_provider_cache_write_input_tokens,
                quantilesTDigestMerge({qs})(processing_time_ms_quantiles) as processing_time_ms_quantiles,
                quantilesTDigestMerge({qs})(ttft_ms_quantiles) as ttft_ms_quantiles
            FROM VariantStatistics
            WHERE {where_clause}
            GROUP BY function_name, variant_name
            ORDER BY function_name, variant_name
            FORMAT JSONEachRow
            "
        );

        let response = self.run_query_synchronous(query, &query_params).await?;

        if response.response.is_empty() {
            return Ok(vec![]);
        }

        let rows: Vec<VariantStatisticsRow> = response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to parse VariantStatisticsRow: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    fn get_variant_statistics_quantiles(&self) -> Option<&[f64]> {
        Some(QUANTILES)
    }
}
