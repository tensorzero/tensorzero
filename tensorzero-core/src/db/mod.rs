use std::future::Future;

use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::error::Error;

pub mod clickhouse;

pub trait DatabaseConnection: SelectQueries + HealthCheckable {}

pub trait HealthCheckable {
    fn health(&self) -> impl Future<Output = Result<(), Error>>;
}

pub trait SelectQueries {
    fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> impl Future<Output = Result<Vec<ModelUsageTimePoint>, Error>>;
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

impl<T: SelectQueries + HealthCheckable> DatabaseConnection for T {}
