use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_derive::export_schema;
use uuid::Uuid;

use crate::db::inferences::{
    InferenceOutputSource, ListInferencesParams, DEFAULT_INFERENCE_QUERY_LIMIT,
};
use crate::stored_inference::StoredInference;

// Re-exported for backwards compatibility.
pub use crate::endpoints::shared_types::OrderDirection;

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
pub struct FloatMetricFilter {
    pub metric_name: String,
    pub value: f64,
    pub comparison_operator: FloatComparisonOperator,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
pub struct BooleanMetricFilter {
    pub metric_name: String,
    pub value: bool,
}

/// Filter by tag key-value pair.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
pub struct TagFilter {
    pub key: String,
    pub value: String,
    pub comparison_operator: TagComparisonOperator,
}

/// Filter by timestamp.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
pub struct TimeFilter {
    #[ts(type = "Date")]
    #[schemars(with = "String")]
    pub time: DateTime<Utc>,
    pub comparison_operator: TimeComparisonOperator,
}

/// Comparison operators for float metrics.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, JsonSchema, ts_rs::TS)]
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
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, JsonSchema, ts_rs::TS)]
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
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
pub enum TagComparisonOperator {
    #[serde(rename = "=")]
    Equal,
    #[serde(rename = "!=")]
    NotEqual,
}

/// The property to order by.
/// This is flattened in the public API inside the `OrderBy` struct.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, ts_rs::TS)]
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
    /// Current relevance metric is very rudimentary (just term frequency), but we plan
    /// to improve it in the future.
    #[schemars(title = "OrderBySearchRelevance")]
    SearchRelevance,
}

/// Order by clauses for querying inferences.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, ts_rs::TS)]
#[ts(export)]
pub struct OrderBy {
    /// The property to order by.
    #[serde(flatten)]
    pub term: OrderByTerm,

    /// The ordering direction.
    pub direction: OrderDirection,
}

/// Filters for querying inferences.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
#[export_schema]
pub enum InferenceFilter {
    /// Filter by the value of a float metric
    #[schemars(title = "InferenceFilterFloatMetric")]
    FloatMetric(FloatMetricFilter),

    /// Filter by the value of a boolean metric
    #[schemars(title = "InferenceFilterBooleanMetric")]
    BooleanMetric(BooleanMetricFilter),

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

/// Request to list inferences with pagination and filters.
/// Used by the `POST /v1/inferences/list_inferences` endpoint.
#[derive(Debug, Deserialize, Default, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export, optional_fields)]
#[export_schema]
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

    /// Text query to filter. Case-insensitive substring search over the inferences' input and output.
    ///
    /// THIS FEATURE IS EXPERIMENTAL, and we may change or remove it at any time.
    /// We recommend against depending on this feature for critical use cases.
    ///
    /// Important limitations:
    /// - This requires an exact substring match; we do not tokenize this query string.
    /// - This doesn't search for any content in the template itself.
    /// - Quality is based on term frequency > 0, without any relevance scoring.
    /// - There are no performance guarantees (it's best effort only). Today, with no other
    ///   filters, it will perform a full table scan, which may be extremely slow depending
    ///   on the data volume.
    pub search_query_experimental: Option<String>,
}

impl ListInferencesRequest {
    /// Convert the request to a `ListInferencesParams` struct for the database query layer.
    pub fn as_list_inferences_params<'a>(&'a self) -> ListInferencesParams<'a> {
        ListInferencesParams {
            ids: None,
            function_name: self.function_name.as_deref(),
            variant_name: self.variant_name.as_deref(),
            episode_id: self.episode_id.as_ref(),
            filters: self.filter.as_ref(),
            output_source: self.output_source,
            limit: self.limit.unwrap_or(DEFAULT_INFERENCE_QUERY_LIMIT),
            offset: self.offset.unwrap_or(0),
            order_by: self.order_by.as_deref(),
            search_query_experimental: self.search_query_experimental.as_deref(),
        }
    }
}

/// Request to get specific inferences by their IDs.
/// Used by the `POST /v1/inferences/get_inferences` endpoint.
#[derive(Debug, Deserialize, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export, optional_fields)]
#[export_schema]
pub struct GetInferencesRequest {
    /// The IDs of the inferences to retrieve. Required.
    pub ids: Vec<Uuid>,

    /// Optional function name to filter by.
    /// Including this improves query performance since `function_name` is the first column
    /// in the ClickHouse primary key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_name: Option<String>,

    /// Source of the inference output.
    /// Determines whether to return the original inference output or demonstration feedback
    /// (manually-curated output) if available.
    pub output_source: InferenceOutputSource,
}

/// Response containing the requested inferences.
#[derive(Debug, Deserialize, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
#[export_schema]
pub struct GetInferencesResponse {
    /// The retrieved inferences.
    pub inferences: Vec<StoredInference>,
}

/// Response containing the inference table bounds.
/// Used by the `GET /internal/inferences/bounds` endpoint.
#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[serde_with::skip_serializing_none]
#[ts(export, optional_fields)]
pub struct GetInferenceBoundsResponse {
    /// The most recent inference ID (MAX id_uint).
    pub latest_id: Option<Uuid>,

    /// The oldest inference ID (MIN id_uint).
    pub earliest_id: Option<Uuid>,

    /// The total number of inferences matching the filter criteria.
    pub count: u64,
}

/// Metadata about an inference.
/// Used by the `GET /internal/inferences` endpoint.
#[derive(Debug, Deserialize, Serialize, ts_rs::TS, Clone, PartialEq)]
#[ts(export)]
pub struct InternalInferenceMetadata {
    /// The ID of the inference.
    pub id: Uuid,

    /// The function name of the inference.
    pub function_name: String,

    /// The variant name of the inference.
    pub variant_name: String,

    /// The episode ID of the inference.
    pub episode_id: Uuid,

    /// The function type of the inference.
    pub function_type: String,

    /// The timestamp of the inference.
    pub timestamp: DateTime<Utc>,
}

/// Response containing the list of inferences by ID.
#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct InternalListInferencesByIdResponse {
    /// The list of inferences.
    pub inferences: Vec<InternalInferenceMetadata>,
}
