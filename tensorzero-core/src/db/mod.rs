use async_trait::async_trait;
use chrono::{DateTime, Utc};
use feedback::FeedbackQueries;
use serde::{Deserialize, Serialize};
use sqlx::postgres::types::PgInterval;
use uuid::Uuid;

use crate::db::datasets::DatasetQueries;
use crate::error::Error;
use crate::rate_limiting::ActiveRateLimitKey;
use crate::serde_util::{deserialize_option_u64, deserialize_u64};

pub mod clickhouse;
pub mod datasets;
pub mod feedback;
pub mod inferences;
pub mod postgres;

#[async_trait]
pub trait ClickHouseConnection:
    SelectQueries + DatasetQueries + FeedbackQueries + HealthCheckable + Send + Sync
{
}

#[async_trait]
pub trait PostgresConnection: RateLimitQueries + HealthCheckable + Send + Sync {}

#[async_trait]
pub trait HealthCheckable {
    async fn health(&self) -> Result<(), Error>;
}

#[async_trait]
pub trait SelectQueries {
    async fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<ModelUsageTimePoint>, Error>;

    async fn get_model_latency_quantiles(
        &self,
        time_window: TimeWindow,
    ) -> Result<Vec<ModelLatencyDatapoint>, Error>;

    async fn count_distinct_models_used(&self) -> Result<u32, Error>;

    async fn query_episode_table(
        &self,
        limit: u32,
        before: Option<Uuid>,
        after: Option<Uuid>,
    ) -> Result<Vec<EpisodeByIdRow>, Error>;

    async fn query_episode_table_bounds(&self) -> Result<TableBoundsWithCount, Error>;
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum TimeWindow {
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Cumulative,
}

#[derive(Debug, ts_rs::TS, Serialize, Deserialize, PartialEq)]
#[ts(export)]
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

#[derive(Debug, ts_rs::TS, Serialize, Deserialize, PartialEq)]
#[ts(export)]
pub struct ModelLatencyDatapoint {
    pub model_name: String,
    // should be an array of quantiles_len u64
    pub response_time_ms_quantiles: Vec<Option<f32>>,
    pub ttft_ms_quantiles: Vec<Option<f32>>,
    #[serde(deserialize_with = "deserialize_u64")]
    pub count: u64,
}

#[derive(Debug, ts_rs::TS, Serialize, Deserialize, PartialEq)]
#[ts(export)]
pub struct EpisodeByIdRow {
    pub episode_id: Uuid,
    #[serde(deserialize_with = "deserialize_u64")]
    pub count: u64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub last_inference_id: Uuid,
}

#[derive(Debug, ts_rs::TS, Serialize, Deserialize, PartialEq)]
#[ts(export)]
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

pub trait RateLimitQueries {
    /// This function will fail if any of the requests individually fail.
    /// It is an atomic operation so no tickets will be consumed if any request fails.
    async fn consume_tickets(
        &self,
        requests: &[ConsumeTicketsRequest],
    ) -> Result<Vec<ConsumeTicketsReceipt>, Error>;

    async fn return_tickets(
        &self,
        requests: Vec<ReturnTicketsRequest>,
    ) -> Result<Vec<ReturnTicketsReceipt>, Error>;

    async fn get_balance(
        &self,
        key: &str,
        capacity: u64,
        refill_amount: u64,
        refill_interval: PgInterval,
    ) -> Result<u64, Error>;
}

#[derive(Debug)]
pub struct ConsumeTicketsRequest {
    pub key: ActiveRateLimitKey,
    pub requested: u64,
    pub capacity: u64,
    pub refill_amount: u64,
    pub refill_interval: PgInterval,
}

#[derive(Debug)]
pub struct ConsumeTicketsReceipt {
    pub key: ActiveRateLimitKey,
    pub success: bool,
    pub tickets_remaining: u64,
    pub tickets_consumed: u64,
}

pub struct ReturnTicketsRequest {
    pub key: ActiveRateLimitKey,
    pub returned: u64,
    pub capacity: u64,
    pub refill_amount: u64,
    pub refill_interval: PgInterval,
}

pub struct ReturnTicketsReceipt {
    pub key: ActiveRateLimitKey,
    pub balance: u64,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct TableBounds {
    #[ts(optional)]
    pub first_id: Option<Uuid>,
    #[ts(optional)]
    pub last_id: Option<Uuid>,
}

impl<T: RateLimitQueries + ExperimentationQueries + HealthCheckable + Send + Sync>
    PostgresConnection for T
{
}

pub trait ExperimentationQueries {
    async fn check_and_set_variant_by_episode(
        &self,
        episode_id: Uuid,
        function_name: &str,
        candidate_variant_name: &str,
    ) -> Result<String, Error>;
}
