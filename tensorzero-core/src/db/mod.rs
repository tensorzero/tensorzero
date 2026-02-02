use async_trait::async_trait;
use chrono::{DateTime, Utc};
use feedback::FeedbackQueries;
use serde::{Deserialize, Serialize};
use std::future::Future;
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
use crate::db::datasets::DatasetQueries;
use crate::error::Error;
use crate::serde_util::{deserialize_option_u64, deserialize_u64};

pub mod batch_inference;
pub mod clickhouse;
pub mod datasets;
pub mod delegating_connection;
pub mod evaluation_queries;
pub mod feedback;
pub mod inference_count;
pub mod inferences;
pub mod model_inferences;
pub mod postgres;
pub mod query_helpers;
pub mod rate_limiting;
pub mod stored_datapoint;
pub mod test_helpers;
pub mod valkey;
pub mod workflow_evaluation_queries;

// For backcompat, re-export everything from the rate_limiting module
pub use rate_limiting::*;

#[async_trait]
pub trait ClickHouseConnection:
    SelectQueries + DatasetQueries + FeedbackQueries + HealthCheckable + Send + Sync
{
}

#[async_trait]
pub trait HealthCheckable {
    async fn health(&self) -> Result<(), Error>;
}

#[cfg_attr(test, automock)]
pub trait SelectQueries {
    fn count_distinct_models_used(&self) -> impl Future<Output = Result<u32, Error>> + Send;

    fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> impl Future<Output = Result<Vec<ModelUsageTimePoint>, Error>> + Send;

    fn get_model_latency_quantiles(
        &self,
        time_window: TimeWindow,
    ) -> impl Future<Output = Result<Vec<ModelLatencyDatapoint>, Error>> + Send;

    fn query_episode_table(
        &self,
        limit: u32,
        before: Option<Uuid>,
        after: Option<Uuid>,
    ) -> impl Future<Output = Result<Vec<EpisodeByIdRow>, Error>> + Send;

    fn query_episode_table_bounds(
        &self,
    ) -> impl Future<Output = Result<TableBoundsWithCount, Error>> + Send;
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum TimeWindow {
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Cumulative,
}

impl TimeWindow {
    /// Converts the time window to the ClickHouse interval function string.
    /// Returns the string used in dateTrunc and other time functions.
    pub fn to_clickhouse_string(&self) -> &'static str {
        match self {
            TimeWindow::Minute => "minute",
            TimeWindow::Hour => "hour",
            TimeWindow::Day => "day",
            TimeWindow::Week => "week",
            TimeWindow::Month => "month",
            TimeWindow::Cumulative => "year", // Cumulative uses a full year as fallback
        }
    }

    /// Converts the time window to the PostgreSQL date_trunc time unit.
    pub fn to_postgres_time_unit(&self) -> &'static str {
        match self {
            TimeWindow::Minute => "minute",
            TimeWindow::Hour => "hour",
            TimeWindow::Day => "day",
            TimeWindow::Week => "week",
            TimeWindow::Month => "month",
            TimeWindow::Cumulative => "year", // Not used, but uses a full year as fallback
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ModelUsageTimePoint {
    pub period_start: DateTime<Utc>,
    pub model_name: String,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub input_tokens: Option<u64>,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub output_tokens: Option<u64>,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub count: Option<u64>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ModelLatencyDatapoint {
    pub model_name: String,
    // should be an array of quantiles_len u64
    pub response_time_ms_quantiles: Vec<Option<f32>>,
    pub ttft_ms_quantiles: Vec<Option<f32>>,
    #[serde(deserialize_with = "deserialize_u64")]
    pub count: u64,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EpisodeByIdRow {
    pub episode_id: Uuid,
    #[serde(deserialize_with = "deserialize_u64")]
    pub count: u64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub last_inference_id: Uuid,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct TableBoundsWithCount {
    pub first_id: Option<Uuid>,
    pub last_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_u64")]
    pub count: u64,
}

impl<T: SelectQueries + DatasetQueries + FeedbackQueries + HealthCheckable + Send + Sync>
    ClickHouseConnection for T
{
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct TableBounds {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<Uuid>,
}

pub trait ExperimentationQueries {
    async fn check_and_set_variant_by_episode(
        &self,
        episode_id: Uuid,
        function_name: &str,
        candidate_variant_name: &str,
    ) -> Result<String, Error>;
}

#[cfg_attr(test, automock)]
pub trait ConfigQueries {
    fn get_config_snapshot(
        &self,
        snapshot_hash: SnapshotHash,
    ) -> impl Future<Output = Result<ConfigSnapshot, Error>> + Send;
}
