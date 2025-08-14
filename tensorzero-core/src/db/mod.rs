use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::error::Error;

pub mod clickhouse;

#[async_trait]
pub trait DatabaseConnection: SelectQueries + HealthCheckable + Send + Sync {}

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
}

#[derive(Debug)]
pub enum TimeWindow {
    Hour,
    Day,
    Week,
    Month,
    Cumulative,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct ModelUsageTimePoint {
    pub period_start: DateTime<Utc>,
    pub model_name: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub count: u64,
}

impl<T: SelectQueries + HealthCheckable + Send + Sync> DatabaseConnection for T {}
