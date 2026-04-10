//! Variant statistics query trait and types.
//!
//! Provides aggregated usage and cost statistics per (function_name, variant_name)
//! from the `VariantStatistics` materialized view (ClickHouse) or rollup table (Postgres).

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::serde_util::{deserialize_option_u64, deserialize_u64};

/// Query parameters for variant statistics.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetVariantStatisticsParams {
    /// The function name to query statistics for.
    pub function_name: String,
    /// Optional filter for specific variants. If not provided, all variants are included.
    #[serde(default)]
    pub variant_names: Option<Vec<String>>,
    /// Optional lower bound on the time window (inclusive).
    #[serde(default)]
    pub after: Option<DateTime<Utc>>,
    /// Optional upper bound on the time window (exclusive).
    #[serde(default)]
    pub before: Option<DateTime<Utc>>,
}

/// A single row of aggregated variant statistics.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct VariantStatisticsRow {
    pub function_name: String,
    pub variant_name: String,
    #[serde(deserialize_with = "deserialize_u64")]
    pub inference_count: u64,
    #[serde(default, deserialize_with = "deserialize_option_u64")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_option_u64")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_output_tokens: Option<u64>,
    #[serde(default, with = "rust_decimal::serde::float_option")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(type = "number | null"))]
    pub total_cost: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_option_u64")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_with_cost: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_option_u64")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_provider_cache_read_input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_option_u64")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_provider_cache_write_input_tokens: Option<u64>,
    /// Latency quantiles — only populated from ClickHouse, None on Postgres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processing_time_ms_quantiles: Option<Vec<Option<f32>>>,
    /// TTFT quantiles — only populated from ClickHouse, None on Postgres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms_quantiles: Option<Vec<Option<f32>>>,
}

/// Response type for the variant statistics endpoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct GetVariantStatisticsResponse {
    /// The quantile inputs (e.g. [0.001, 0.005, ..., 0.999]) — populated when ClickHouse
    /// is the backend, None on Postgres.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantiles: Option<Vec<f64>>,
    /// The aggregated variant statistics rows.
    pub data: Vec<VariantStatisticsRow>,
}

/// Trait for querying variant statistics.
#[async_trait]
pub trait VariantStatisticsQueries: Send + Sync {
    /// Query aggregated variant statistics, optionally filtered by variant names
    /// and a lower time bound.
    async fn get_variant_statistics(
        &self,
        params: &GetVariantStatisticsParams,
    ) -> Result<Vec<VariantStatisticsRow>, Error>;

    /// Returns the quantile levels used for latency distributions, if available.
    ///
    /// ClickHouse returns the QUANTILES constant; Postgres returns None since
    /// latency quantiles are not available from the rollup table.
    fn get_variant_statistics_quantiles(&self) -> Option<&[f64]>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::prelude::FromPrimitive;

    #[test]
    fn test_variant_statistics_row_serde_full() {
        let row = VariantStatisticsRow {
            function_name: "my_function".to_string(),
            variant_name: "variant_a".to_string(),
            inference_count: 42,
            total_input_tokens: Some(1000),
            total_output_tokens: Some(500),
            total_cost: Decimal::from_f64(0.00123),
            count_with_cost: Some(40),
            total_provider_cache_read_input_tokens: Some(100),
            total_provider_cache_write_input_tokens: Some(200),
            processing_time_ms_quantiles: Some(vec![Some(10.0), Some(50.0), Some(100.0)]),
            ttft_ms_quantiles: Some(vec![Some(5.0), Some(25.0), Some(50.0)]),
        };

        let json = serde_json::to_string(&row).expect("serialization should succeed");
        assert!(
            json.contains("\"total_cost\":0.00123"),
            "Cost should be serialized as float: {json}"
        );
        assert!(json.contains("\"inference_count\":42"));

        let deserialized: VariantStatisticsRow =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(deserialized, row);
    }

    #[test]
    fn test_variant_statistics_row_serde_minimal() {
        // Simulates a row with no optional fields (Postgres-like, no quantiles)
        let json = r#"{"function_name":"f","variant_name":"v","inference_count":"10"}"#;
        let row: VariantStatisticsRow =
            serde_json::from_str(json).expect("deserialization should succeed");
        assert_eq!(row.function_name, "f");
        assert_eq!(row.variant_name, "v");
        assert_eq!(row.inference_count, 10);
        assert_eq!(row.total_input_tokens, None);
        assert_eq!(row.total_cost, None);
        assert_eq!(row.processing_time_ms_quantiles, None);
    }

    #[test]
    fn test_variant_statistics_row_omits_none_fields() {
        let row = VariantStatisticsRow {
            function_name: "f".to_string(),
            variant_name: "v".to_string(),
            inference_count: 1,
            total_input_tokens: None,
            total_output_tokens: None,
            total_cost: None,
            count_with_cost: None,
            total_provider_cache_read_input_tokens: None,
            total_provider_cache_write_input_tokens: None,
            processing_time_ms_quantiles: None,
            ttft_ms_quantiles: None,
        };

        let json = serde_json::to_string(&row).expect("serialization should succeed");
        assert!(
            !json.contains("total_input_tokens"),
            "None fields should be omitted: {json}"
        );
        assert!(
            !json.contains("total_cost"),
            "None fields should be omitted: {json}"
        );
        assert!(
            !json.contains("processing_time_ms_quantiles"),
            "None fields should be omitted: {json}"
        );
    }

    #[test]
    fn test_get_variant_statistics_response_serde() {
        let response = GetVariantStatisticsResponse {
            quantiles: Some(vec![0.5, 0.9, 0.99]),
            data: vec![VariantStatisticsRow {
                function_name: "f".to_string(),
                variant_name: "v".to_string(),
                inference_count: 100,
                total_input_tokens: Some(5000),
                total_output_tokens: Some(2000),
                total_cost: Decimal::from_f64(1.5),
                count_with_cost: Some(100),
                total_provider_cache_read_input_tokens: None,
                total_provider_cache_write_input_tokens: None,
                processing_time_ms_quantiles: None,
                ttft_ms_quantiles: None,
            }],
        };

        let json = serde_json::to_string(&response).expect("serialization should succeed");
        assert!(json.contains("\"quantiles\":[0.5,0.9,0.99]"));
        assert!(json.contains("\"inference_count\":100"));

        let deserialized: GetVariantStatisticsResponse =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(deserialized.quantiles, Some(vec![0.5, 0.9, 0.99]));
        assert_eq!(deserialized.data.len(), 1);
    }

    #[test]
    fn test_get_variant_statistics_response_no_quantiles() {
        let response = GetVariantStatisticsResponse {
            quantiles: None,
            data: vec![],
        };

        let json = serde_json::to_string(&response).expect("serialization should succeed");
        assert!(
            !json.contains("quantiles"),
            "None quantiles should be omitted: {json}"
        );
    }
}
