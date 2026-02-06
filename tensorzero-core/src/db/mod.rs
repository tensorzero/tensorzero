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

    /// Converts the time window to the Postgres date_trunc time unit.
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
#[derive(Debug, Serialize, Deserialize, PartialEq, sqlx::FromRow)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EpisodeByIdRow {
    pub episode_id: Uuid,
    #[serde(deserialize_with = "deserialize_u64")]
    #[sqlx(try_from = "i64")]
    pub count: u64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub last_inference_id: Uuid,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq, sqlx::FromRow)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct TableBoundsWithCount {
    pub first_id: Option<Uuid>,
    pub last_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_u64")]
    #[sqlx(try_from = "i64")]
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

#[async_trait]
#[cfg_attr(test, automock)]
pub trait ConfigQueries: Send + Sync {
    async fn get_config_snapshot(
        &self,
        snapshot_hash: SnapshotHash,
    ) -> Result<ConfigSnapshot, Error>;

    async fn write_config_snapshot(&self, snapshot: &ConfigSnapshot) -> Result<(), Error>;
}

/// A stored DICL (Dynamic In-Context Learning) example.
#[derive(Debug, Clone)]
pub struct StoredDICLExample {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub namespace: String,
    pub input: String,
    pub output: String,
    pub embedding: Vec<f32>,
    pub created_at: DateTime<Utc>,
}

/// A DICL example returned from similarity search.
#[derive(Debug, Clone)]
pub struct DICLExampleWithDistance {
    pub input: String,
    pub output: String,
    pub cosine_distance: f32,
}

/// Trait for DICL (Dynamic In-Context Learning) queries.
///
/// DICL stores examples with embeddings for similarity search during inference.
/// The variant retrieves similar examples based on the input embedding to provide
/// in-context learning examples to the model.
pub trait DICLQueries {
    /// Insert a DICL example into the database.
    fn insert_dicl_example(
        &self,
        example: &StoredDICLExample,
    ) -> impl Future<Output = Result<(), Error>> + Send;

    /// Insert multiple DICL examples in a batch.
    fn insert_dicl_examples(
        &self,
        examples: &[StoredDICLExample],
    ) -> impl Future<Output = Result<u64, Error>> + Send;

    /// Get similar DICL examples using cosine distance.
    ///
    /// Returns examples sorted by cosine distance (ascending).
    fn get_similar_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
        embedding: &[f32],
        limit: u32,
    ) -> impl Future<Output = Result<Vec<DICLExampleWithDistance>, Error>> + Send;

    /// Check if DICL examples exist for a given function and variant.
    fn has_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
    ) -> impl Future<Output = Result<bool, Error>> + Send;

    /// Delete DICL examples for a given function and variant.
    ///
    /// If namespace is provided, only deletes examples in that namespace.
    fn delete_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
        namespace: Option<&str>,
    ) -> impl Future<Output = Result<u64, Error>> + Send;
}
