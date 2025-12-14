//! Types for model statistics endpoints.

use serde::{Deserialize, Serialize};

use crate::db::{ModelLatencyDatapoint, ModelUsageTimePoint, TimeWindow};

/// Response containing the count of distinct models used.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
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
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
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

/// Response containing model latency quantile data.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetModelLatencyResponse {
    /// The model latency data points with quantile distributions.
    pub data: Vec<ModelLatencyDatapoint>,
}
