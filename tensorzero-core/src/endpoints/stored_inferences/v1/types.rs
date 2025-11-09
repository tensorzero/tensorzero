use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::inferences::InferenceOutputSource;
use crate::stored_inference::StoredInference;

#[derive(Debug, Clone, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct FloatMetricFilter {
    pub metric_name: String,
    pub value: f64,
    pub comparison_operator: FloatComparisonOperator,
}

#[derive(Debug, Clone, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct BooleanMetricFilter {
    pub metric_name: String,
    pub value: bool,
}

/// Filter by tag key-value pair.
#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TagFilter {
    pub key: String,
    pub value: String,
    pub comparison_operator: TagComparisonOperator,
}

/// Filter by timestamp.
#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TimeFilter {
    #[ts(type = "Date")]
    pub time: DateTime<Utc>,
    pub comparison_operator: TimeComparisonOperator,
}

/// Comparison operators for float metrics.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
pub enum FloatComparisonOperator {
    #[serde(rename = "<")]
    LessThan,
    #[serde(rename = "<=")]
    LessThanOrEqual,
    #[serde(rename = "=")]
    Equal,
    #[serde(rename = ">")]
    GreaterThan,
    #[serde(rename = ">=")]
    GreaterThanOrEqual,
    #[serde(rename = "!=")]
    NotEqual,
}

/// Comparison operators for timestamps.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
pub enum TimeComparisonOperator {
    #[serde(rename = "<")]
    LessThan,
    #[serde(rename = "<=")]
    LessThanOrEqual,
    #[serde(rename = "=")]
    Equal,
    #[serde(rename = ">")]
    GreaterThan,
    #[serde(rename = ">=")]
    GreaterThanOrEqual,
    #[serde(rename = "!=")]
    NotEqual,
}

/// Comparison operators for tag filters.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
pub enum TagComparisonOperator {
    #[serde(rename = "=")]
    Equal,
    #[serde(rename = "!=")]
    NotEqual,
}

/// The ordering direction.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS)]
#[ts(export)]
pub enum OrderDirection {
    #[serde(rename = "ascending")]
    Asc,
    #[serde(rename = "descending")]
    Desc,
}

/// The property to order by.
/// This is flattened in the public API inside the `OrderBy` struct.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum OrderByTerm {
    Timestamp,
    Metric {
        /// The name of the metric to order by.
        name: String,
    },
}

/// Order by clauses for querying inferences.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS)]
#[ts(export)]
pub struct OrderBy {
    /// The property to order by.
    #[serde(flatten)]
    pub term: OrderByTerm,

    /// The ordering direction.
    pub direction: OrderDirection,
}

/// Filters for querying inferences.
#[derive(Debug, Clone, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceFilter {
    /// Filter by the value of a float metric
    FloatMetric(FloatMetricFilter),

    /// Filter by the value of a boolean metric
    BooleanMetric(BooleanMetricFilter),

    /// Filter by tag key-value pair
    Tag(TagFilter),

    /// Filter by the timestamp of an inference.
    Time(TimeFilter),

    /// Logical AND of multiple filters
    And { children: Vec<InferenceFilter> },

    /// Logical OR of multiple filters
    Or { children: Vec<InferenceFilter> },

    /// Logical NOT of a filter
    Not { child: Box<InferenceFilter> },
}

/// Request to list inferences with pagination and filters.
/// Used by the `POST /v1/inferences/list_inferences` endpoint.
#[derive(Debug, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct ListInferencesRequest {
    /// Optional function name to filter inferences by.
    /// If provided, only inferences from this function will be returned.
    pub function_name: Option<String>,

    /// Optional variant name to filter inferences by.
    /// If provided, only inferences from this variant will be returned.
    pub variant_name: Option<String>,

    /// Optional episode ID to filter inferences by.
    /// If provided, only inferences from this episode will be returned.
    pub episode_id: Option<Uuid>,

    /// Source of the inference output. Determines whether to return the original
    /// inference output or demonstration feedback (manually-curated output) if available.
    pub output_source: InferenceOutputSource,

    /// The maximum number of inferences to return.
    /// Defaults to 20.
    pub limit: Option<u32>,

    /// The number of inferences to skip before starting to return results.
    /// Defaults to 0.
    pub offset: Option<u32>,

    /// Optional filter to apply when querying inferences.
    /// Supports filtering by metrics, tags, time, and logical combinations (AND/OR/NOT).
    pub filter: Option<InferenceFilter>,

    /// Optional ordering criteria for the results.
    /// Supports multiple sort criteria (e.g., sort by timestamp then by metric).
    pub order_by: Option<Vec<OrderBy>>,
}

/// Request to get specific inferences by their IDs.
/// Used by the `POST /v1/inferences/get_inferences` endpoint.
#[derive(Debug, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetInferencesRequest {
    /// The IDs of the inferences to retrieve. Required.
    pub ids: Vec<Uuid>,

    /// Source of the inference output.
    /// Determines whether to return the original inference output or demonstration feedback
    /// (manually-curated output) if available.
    pub output_source: InferenceOutputSource,
}

/// Response containing the requested inferences.
#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GetInferencesResponse {
    /// The retrieved inferences.
    pub inferences: Vec<StoredInference>,
}
