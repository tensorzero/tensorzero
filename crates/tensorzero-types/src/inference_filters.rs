//! Filter and ordering types for querying inferences.
//!
//! These types were originally in `tensorzero-core::endpoints::stored_inferences`
//! and `tensorzero-core::endpoints::shared_types` but are needed by the `db` module,
//! creating a circular dependency. Moving them here breaks that cycle.

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;
use tensorzero_derive::export_schema;

/// The ordering direction.
#[derive(ts_rs::TS, Copy, Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
#[ts(export)]
pub enum OrderDirection {
    #[serde(rename = "ascending")]
    Asc,
    #[serde(rename = "descending")]
    Desc,
}

impl OrderDirection {
    pub fn to_sql_direction(&self) -> &str {
        match self {
            OrderDirection::Asc => "ASC",
            OrderDirection::Desc => "DESC",
        }
    }

    pub fn inverted(&self) -> Self {
        match self {
            OrderDirection::Asc => OrderDirection::Desc,
            OrderDirection::Desc => OrderDirection::Asc,
        }
    }
}

#[derive(ts_rs::TS, Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[ts(export)]
pub struct FloatMetricFilter {
    pub metric_name: String,
    pub value: f64,
    pub comparison_operator: FloatComparisonOperator,
}

#[derive(ts_rs::TS, Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[ts(export)]
pub struct BooleanMetricFilter {
    pub metric_name: String,
    pub value: bool,
}

/// Filter by tag key-value pair.
#[derive(ts_rs::TS, Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[ts(export)]
pub struct TagFilter {
    pub key: String,
    pub value: String,
    pub comparison_operator: TagComparisonOperator,
}

/// Filter by timestamp.
#[derive(ts_rs::TS, Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[ts(export)]
pub struct TimeFilter {
    #[ts(type = "Date")]
    #[schemars(with = "String")]
    pub time: DateTime<Utc>,
    pub comparison_operator: TimeComparisonOperator,
}

/// Comparison operators for float metrics.
#[derive(ts_rs::TS, Clone, Copy, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
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

impl FloatComparisonOperator {
    pub fn to_sql_operator(&self) -> &str {
        match self {
            FloatComparisonOperator::LessThan => "<",
            FloatComparisonOperator::LessThanOrEqual => "<=",
            FloatComparisonOperator::Equal => "=",
            FloatComparisonOperator::GreaterThan => ">",
            FloatComparisonOperator::GreaterThanOrEqual => ">=",
            FloatComparisonOperator::NotEqual => "!=",
        }
    }
}

/// Comparison operators for timestamps.
#[derive(ts_rs::TS, Clone, Copy, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
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

impl TimeComparisonOperator {
    pub fn to_sql_operator(&self) -> &str {
        match self {
            TimeComparisonOperator::LessThan => "<",
            TimeComparisonOperator::LessThanOrEqual => "<=",
            TimeComparisonOperator::Equal => "=",
            TimeComparisonOperator::GreaterThan => ">",
            TimeComparisonOperator::GreaterThanOrEqual => ">=",
            TimeComparisonOperator::NotEqual => "!=",
        }
    }
}

/// Comparison operators for tag filters.
#[derive(ts_rs::TS, Clone, Copy, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[ts(export)]
pub enum TagComparisonOperator {
    #[serde(rename = "=")]
    Equal,
    #[serde(rename = "!=")]
    NotEqual,
}

impl TagComparisonOperator {
    pub fn to_sql_operator(&self) -> &str {
        match self {
            TagComparisonOperator::Equal => "=",
            TagComparisonOperator::NotEqual => "!=",
        }
    }
}

/// Filter by whether an inference has a demonstration.
#[derive(ts_rs::TS, Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[ts(export)]
pub struct DemonstrationFeedbackFilter {
    pub has_demonstration: bool,
}

/// The property to order by.
/// This is flattened in the public API inside the `OrderBy` struct.
#[derive(ts_rs::TS, Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
#[ts(export)]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum OrderByTerm {
    // These titles become the names of the top-level OrderBy structs in the generated
    // schema, because it's flattened.
    /// Creation timestamp of the item.
    #[schemars(title = "OrderByTimestamp")]
    Timestamp,

    /// Value of a metric.
    #[schemars(title = "OrderByMetric")]
    Metric {
        /// The name of the metric to order by.
        name: String,
    },

    /// Relevance score of the search query in the input and output of the item.
    /// Requires a search query (experimental). If it's not provided, we return an error.
    ///
    /// NOTE: Relevance ordering is not yet implemented for Postgres and currently
    /// falls back to id ordering. See TODO(#6441).
    #[schemars(title = "OrderBySearchRelevance")]
    SearchRelevance,
}

/// Order by clauses for querying inferences.
#[derive(ts_rs::TS, Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
#[ts(export)]
pub struct OrderBy {
    /// The property to order by.
    #[serde(flatten)]
    pub term: OrderByTerm,

    /// The ordering direction.
    pub direction: OrderDirection,
}

/// Filters for querying inferences.
/// NOTE: This is stored in DB. Please do not make breaking changes.
/// If we have to, we should make a separate storage type and convert.
#[derive(ts_rs::TS, Clone, Debug, JsonSchema, Serialize, TensorZeroDeserialize)]
#[ts(export)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[export_schema]
pub enum InferenceFilter {
    /// Filter by the value of a float metric
    #[schemars(title = "InferenceFilterFloatMetric")]
    FloatMetric(FloatMetricFilter),

    /// Filter by the value of a boolean metric
    #[schemars(title = "InferenceFilterBooleanMetric")]
    BooleanMetric(BooleanMetricFilter),

    /// Filter by whether an inference has a demonstration.
    #[schemars(title = "InferenceFilterDemonstrationFeedback")]
    DemonstrationFeedback(DemonstrationFeedbackFilter),

    /// Filter by tag key-value pair
    #[schemars(title = "InferenceFilterTag")]
    Tag(TagFilter),

    /// Filter by the timestamp of an inference.
    #[schemars(title = "InferenceFilterTime")]
    Time(TimeFilter),

    /// Logical AND of multiple filters
    #[schemars(title = "InferenceFilterAnd")]
    And { children: Vec<InferenceFilter> },

    /// Logical OR of multiple filters
    #[schemars(title = "InferenceFilterOr")]
    Or { children: Vec<InferenceFilter> },

    /// Logical NOT of a filter
    #[schemars(title = "InferenceFilterNot")]
    Not { child: Box<InferenceFilter> },
}
