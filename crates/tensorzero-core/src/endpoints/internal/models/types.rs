//! Types for model statistics endpoints.

use serde::{Deserialize, Serialize};

use crate::db::{
    CacheStatisticsTimePoint, ModelLatencyDatapoint, ModelUsageTimePoint, TimeWindow,
    VariantUsageTimePoint,
};

/// Response containing the count of distinct models used.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CountModelsResponse {
    /// The count of distinct models used.
    pub model_count: u32,
}

/// Query parameters for the model usage timeseries endpoint.
#[derive(Debug, Deserialize)]
pub struct GetModelUsageQueryParams {
    /// The time window granularity for grouping data.
    pub time_window: TimeWindow,
    /// Maximum number of time periods to return.
    pub max_periods: u32,
}

/// Response containing model usage timeseries data.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetModelUsageResponse {
    /// The model usage data points.
    pub data: Vec<ModelUsageTimePoint>,
}

/// Query parameters for the model latency quantiles endpoint.
#[derive(Debug, Deserialize)]
pub struct GetModelLatencyQueryParams {
    /// The time window for aggregating latency data.
    pub time_window: TimeWindow,
}

/// Query parameters for the cache statistics endpoint.
#[derive(Debug, Deserialize)]
pub struct GetCacheStatisticsQueryParams {
    /// The time window granularity for grouping data.
    pub time_window: TimeWindow,
    /// Maximum number of time periods to return.
    pub max_periods: u32,
    /// Filter by model name (optional).
    pub model_name: Option<String>,
    /// Filter by model provider name (optional).
    pub model_provider_name: Option<String>,
}

/// Response containing cache statistics timeseries data.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetCacheStatisticsResponse {
    /// The cache statistics data points.
    pub data: Vec<CacheStatisticsTimePoint>,
}

/// Query parameters for the variant usage timeseries endpoint.
#[derive(Debug, Deserialize)]
pub struct GetVariantUsageQueryParams {
    /// The time window granularity for grouping data.
    pub time_window: TimeWindow,
    /// Maximum number of time periods to return.
    pub max_periods: u32,
}

/// Response containing variant usage timeseries data.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct GetVariantUsageResponse {
    /// The quantile inputs (e.g. [0.001, 0.005, ..., 0.999]) — populated when ClickHouse
    /// is the backend, None on Postgres.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantiles: Option<Vec<f64>>,
    /// The variant usage data points.
    pub data: Vec<VariantUsageTimePoint>,
}

/// Response containing model latency quantile data.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetModelLatencyResponse {
    /// The quantile inputs (e.g. [0.001, 0.005, ..., 0.999]) used to compute the distributions.
    pub quantiles: Vec<f64>,
    /// The model latency data points with quantile distributions.
    pub data: Vec<ModelLatencyDatapoint>,
}
