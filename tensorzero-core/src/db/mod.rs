use crate::config::rate_limiting::ActiveRateLimitKey;
use crate::error::Error;
use crate::serde_util::{deserialize_option_u64, deserialize_u64};
use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

pub mod clickhouse;
pub mod postgres;

#[async_trait]
pub trait ClickHouseConnection: SelectQueries + HealthCheckable + Send + Sync {}

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
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum TimeWindow {
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

impl<T: SelectQueries + HealthCheckable + Send + Sync> ClickHouseConnection for T {}

pub trait RateLimitQueries {
    /// This function will fail if any of the requests individually fail.
    /// It is an atomic operation so no tickets will be consumed if any request fails.
    async fn consume_tickets(
        &self,
        requests: &[ConsumeTicketsRequest],
    ) -> Result<Vec<ConsumeTicketsReciept>, Error>;

    async fn return_tickets(
        &self,
        requests: &[ReturnTicketsRequest],
    ) -> Result<Vec<ReturnTicketsResult>, Error>;

    async fn get_balance(
        &self,
        key: &str,
        capacity: u64,
        refill_amount: u64,
        refill_interval: TimeDelta,
    ) -> Result<u64, Error>;
}

pub struct ConsumeTicketsRequest {
    pub key: ActiveRateLimitKey,
    pub requested: u64,
    pub capacity: u64,
    pub refill_amount: u64,
    pub refill_interval: TimeDelta,
}

pub struct ConsumeTicketsReciept {
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
    pub refill_interval: TimeDelta,
}

pub struct ReturnTicketsResult {
    pub key: ActiveRateLimitKey,
    pub balance: u64,
}

impl<T: RateLimitQueries + HealthCheckable + Send + Sync> PostgresConnection for T {}
