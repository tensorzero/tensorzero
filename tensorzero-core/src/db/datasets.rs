use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

use crate::config::{MetricConfigLevel, MetricConfigType};
use crate::db::clickhouse::query_builder::{DatapointFilter, FloatComparisonOperator};
use crate::endpoints::datasets::v1::types::DatapointOrderBy;
use crate::endpoints::datasets::{DatapointKind, StoredDatapoint};
use crate::error::Error;
use crate::inference::types::{ContentBlockChatOutput, JsonInferenceOutput, StoredInput};
use crate::serde_util::{
    deserialize_optional_string_or_parsed_json, deserialize_string_or_parsed_json,
    serialize_none_as_empty_map,
};
use crate::tool::{deserialize_optional_tool_info, ToolCallConfigDatabaseInsert};

/// Datapoint types that are directly serialized and inserted into ClickHouse.
/// These should be internal-only types but are exposed to tensorzero-node.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DatapointInsert {
    Chat(ChatInferenceDatapointInsert),
    Json(JsonInferenceDatapointInsert),
}

impl DatapointInsert {
    pub fn id(&self) -> Uuid {
        match self {
            DatapointInsert::Chat(datapoint) => datapoint.id,
            DatapointInsert::Json(datapoint) => datapoint.id,
        }
    }
}

/// Type that gets serialized directly to be written to ClickHouse. Serialization should match
/// the structure of the ChatInferenceDatapoint table in ClickHouse.
/// Theis should be an internal-only type, but it's exposed to tensorzero-node.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatInferenceDatapointInsert {
    /// Name of the dataset to write to. Required.
    pub dataset_name: String,

    /// Name of the function that generated this datapoint. Required.
    pub function_name: String,

    /// Human-readable name of the datapoint. Optional.
    #[serde(default)]
    pub name: Option<String>,

    /// Unique identifier for the datapoint. Required.
    pub id: Uuid,

    /// Episode ID that the datapoint belongs to. Optional.
    #[serde(default)]
    pub episode_id: Option<Uuid>,

    /// Input to the function that generated this datapoint. Required.
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,

    /// Output of the function that generated this datapoint. Optional.
    /// TODO(#4405): this should be a new type StoredContentBlockChatOutput that takes the storage ToolCallOutput format.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_string_or_parsed_json"
    )]
    pub output: Option<Vec<ContentBlockChatOutput>>,

    /// Tool parameters used to generate this datapoint. Optional.
    #[serde(flatten, deserialize_with = "deserialize_optional_tool_info")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,

    /// Tags associated with this datapoint. Optional.
    #[serde(default, serialize_with = "serialize_none_as_empty_map")]
    pub tags: Option<HashMap<String, String>>,

    /// Deprecated, do not use.
    pub auxiliary: String,

    /// Timestamp when the datapoint was marked as stale. Optional.
    #[serde(default)]
    pub staled_at: Option<String>,

    /// Source inference ID that generated this datapoint. Optional.
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,

    /// If true, this datapoint was manually created or edited by the user.
    pub is_custom: bool,
}

/// Type that gets serialized directly to be written to ClickHouse. Serialization should match
/// the structure of the JsonInferenceDatapoint table in ClickHouse.
/// Theis should be an internal-only type, but it's exposed to tensorzero-node.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonInferenceDatapointInsert {
    /// Name of the dataset to write to. Required.
    pub dataset_name: String,

    /// Name of the function that generated this datapoint. Required.
    pub function_name: String,

    /// Human-readable name of the datapoint. Optional.
    #[serde(default)]
    pub name: Option<String>,

    /// Unique identifier for the datapoint. Required.
    pub id: Uuid,

    /// Episode ID that the datapoint belongs to. Optional.
    #[serde(default)]
    pub episode_id: Option<Uuid>,

    /// Input to the function that generated this datapoint. Required.
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,

    /// Output of the function that generated this datapoint. Optional.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_string_or_parsed_json"
    )]
    pub output: Option<JsonInferenceOutput>,

    /// Schema of the output of the function that generated this datapoint. Required.
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output_schema: serde_json::Value,

    /// Tags associated with this datapoint. Optional.
    #[serde(default, serialize_with = "serialize_none_as_empty_map")]
    pub tags: Option<HashMap<String, String>>,

    /// Deprecated, do not use.
    pub auxiliary: String,

    /// Timestamp when the datapoint was marked as stale. Optional.
    #[serde(default)]
    pub staled_at: Option<String>,

    /// Source inference ID that generated this datapoint. Optional.
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,

    /// If true, this datapoint was manually created or edited by the user.
    pub is_custom: bool,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct MetricFilter {
    pub metric: String,
    pub metric_type: MetricConfigType,
    pub operator: FloatComparisonOperator,
    pub threshold: f64,
    pub join_on: MetricConfigLevel,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum DatasetOutputSource {
    // When generating a dataset, don't include any output.
    None,
    // When generating a dataset, include original inference output.
    Inference,
    // When generating a dataset, include any linked demonstration as output.
    Demonstration,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct DatasetQueryParams {
    pub inference_type: DatapointKind,
    pub function_name: Option<String>,
    pub dataset_name: Option<String>,
    pub variant_name: Option<String>,
    pub extra_where: Option<Vec<String>>,
    pub extra_params: Option<HashMap<String, String>>,
    // TODO: consider supporting compound filters (e.g. AND/OR)
    pub metric_filter: Option<MetricFilter>,
    pub output_source: DatasetOutputSource,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}
#[derive(Deserialize, ts_rs::TS)]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct GetDatasetMetadataParams {
    /// Only select datasets matching a specific function.
    pub function_name: Option<String>,

    /// The maximum number of datasets to return.
    pub limit: Option<u32>,

    /// The number of datasets to skip before starting to return results.
    pub offset: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, ts_rs::TS)]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct DatasetMetadata {
    pub dataset_name: String,
    pub count: u32,
    pub last_updated: String,
}

#[derive(Deserialize, ts_rs::TS)]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct CountDatapointsForDatasetFunctionParams {
    pub dataset_name: String,
    pub function_name: String,
    pub function_type: DatapointKind,
}

#[derive(Deserialize, ts_rs::TS)]
#[cfg_attr(test, ts(export, optional_fields))]
/// Legacy struct for old get_datapoint clickhouse query. To be deprecated.
pub struct GetDatapointParams {
    pub dataset_name: String,
    pub datapoint_id: Uuid,
    /// Whether to include stale datapoints in the query; false by default.
    pub allow_stale: Option<bool>,
}

/// A struct representing query params for a SELECT datapoints query.
#[derive(Deserialize)]
pub struct GetDatapointsParams {
    /// Dataset name to query. If not provided, all datasets will be queried.
    /// At least one of `dataset_name` or `ids` must be provided.
    #[serde(default)]
    pub dataset_name: Option<String>,

    /// Function name to filter by. If provided, only datapoints from this function will be returned.
    #[serde(default)]
    pub function_name: Option<String>,

    /// IDs of the datapoints to query. If not provided, all datapoints will be queried.
    /// At least one of `dataset_name` or `ids` must be provided.
    #[serde(default)]
    pub ids: Option<Vec<Uuid>>,

    /// Maximum number of datapoints to return.
    pub limit: u32,

    /// Number of datapoints to skip before starting to return results.
    pub offset: u32,

    /// Whether to include stale datapoints in the query.
    pub allow_stale: bool,

    /// Optional filter to apply when querying datapoints.
    /// Supports filtering by tags, time, and logical combinations (AND/OR/NOT).
    #[serde(default)]
    pub filter: Option<DatapointFilter>,

    /// Optional ordering criteria for the results.
    #[serde(default)]
    pub order_by: Option<Vec<DatapointOrderBy>>,

    /// Text query to filter. Case-insensitive substring search.
    #[serde(default)]
    pub search_query_experimental: Option<String>,
}

#[async_trait]
#[cfg_attr(test, automock)]
pub trait DatasetQueries {
    /// Counts rows for a dataset based on query parameters
    async fn count_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error>;

    /// Inserts rows into a dataset table by selecting from the inference tables
    /// Returns the number of rows inserted
    async fn insert_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error>;

    /// Gets dataset metadata (name, count, last updated)
    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error>;

    /// Gets the count of unique dataset names
    async fn count_datasets(&self) -> Result<u32, Error>;

    /// Inserts a batch of datapoints into the database
    /// Internally separates chat and JSON datapoints and writes them to the appropriate tables
    /// Returns the number of rows written.
    async fn insert_datapoints(&self, datapoints: &[DatapointInsert]) -> Result<u64, Error>;

    /// Counts datapoints for a specific dataset and function
    async fn count_datapoints_for_dataset_function(
        &self,
        params: &CountDatapointsForDatasetFunctionParams,
    ) -> Result<u32, Error>;

    /// Gets a single datapoint by dataset name and ID
    /// TODO(shuyangli): To deprecate in favor of `get_datapoints`
    async fn get_datapoint(&self, params: &GetDatapointParams) -> Result<StoredDatapoint, Error>;

    /// Gets multiple datapoints with various filters and pagination
    async fn get_datapoints(
        &self,
        params: &GetDatapointsParams,
    ) -> Result<Vec<StoredDatapoint>, Error>;

    /// Deletes datapoints or datasets by marking specified datapoints as stale.
    /// This is a soft deletion, so evaluation runs will still refer to it.
    ///
    /// If `datapoint_ids` is None, all datapoints in the dataset will be deleted.
    /// Otherwise, it's required to be non-empty, and only those datapoints with the given IDs will be deleted.
    ///
    /// Returns the number of datapoints that were deleted.
    async fn delete_datapoints(
        &self,
        dataset_name: &str,
        datapoint_ids: Option<&[Uuid]>,
    ) -> Result<u64, Error>;
}
