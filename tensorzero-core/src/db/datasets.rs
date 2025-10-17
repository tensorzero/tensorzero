use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::config::{MetricConfigLevel, MetricConfigType};
use crate::db::clickhouse::query_builder::FloatComparisonOperator;
use crate::endpoints::datasets::{Datapoint, DatapointKind};
use crate::error::Error;
use crate::inference::types::{ContentBlockChatOutput, JsonInferenceOutput, StoredInput};
use crate::serde_util::{
    deserialize_optional_string_or_parsed_json, deserialize_string_or_parsed_json,
};
use crate::tool::ToolCallConfigDatabaseInsert;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum DatapointInsert {
    #[serde(rename = "chat")]
    Chat(ChatInferenceDatapointInsert),
    #[serde(rename = "json")]
    Json(JsonInferenceDatapointInsert),
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct ChatInferenceDatapointInsert {
    pub dataset_name: String,
    pub function_name: String,
    pub name: Option<String>,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
    pub auxiliary: String,
    pub staled_at: Option<String>,
    pub source_inference_id: Option<Uuid>,
    pub is_custom: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct JsonInferenceDatapointInsert {
    pub dataset_name: String,
    pub function_name: String,
    pub name: Option<String>,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    pub output: Option<JsonInferenceOutput>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
    pub auxiliary: String,
    pub staled_at: Option<String>,
    pub source_inference_id: Option<Uuid>,
    pub is_custom: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
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

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
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

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct GetDatasetRowsParams {
    pub dataset_name: String,
    pub page_size: u32,
    pub offset: u32,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct GetDatasetMetadataParams {
    // Only select datasets matching a specific function
    pub function_name: Option<String>,
    pub page_size: Option<u32>,
    pub offset: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct DatasetDetailRow {
    pub id: String,
    #[serde(rename = "type")]
    pub row_type: String,
    pub function_name: String,
    pub name: Option<String>,
    pub episode_id: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct DatasetMetadata {
    pub dataset_name: String,
    pub count: u32,
    pub last_updated: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct AdjacentDatapointIds {
    pub previous_id: Option<Uuid>,
    pub next_id: Option<Uuid>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct StaleDatapointParams {
    pub dataset_name: String,
    pub datapoint_id: Uuid,
    pub function_type: DatapointKind,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct CountDatapointsForDatasetFunctionParams {
    pub dataset_name: String,
    pub function_name: String,
    pub function_type: DatapointKind,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct GetAdjacentDatapointIdsParams {
    pub dataset_name: String,
    pub datapoint_id: Uuid,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct GetDatapointParams {
    pub dataset_name: String,
    pub datapoint_id: Uuid,
    /// Whether to include stale datapoints in the query; false by default.
    pub allow_stale: Option<bool>,
}

#[async_trait]
pub trait DatasetQueries {
    /// Counts rows for a dataset based on query parameters
    async fn count_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error>;

    /// Inserts rows into a dataset table by selecting from the inference tables
    /// Returns the number of rows inserted
    async fn insert_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error>;

    /// Gets rows from a dataset with pagination
    async fn get_dataset_rows(
        &self,
        params: &GetDatasetRowsParams,
    ) -> Result<Vec<DatasetDetailRow>, Error>;

    /// Gets dataset metadata (name, count, last updated)
    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error>;

    /// Gets the count of unique dataset names
    async fn count_datasets(&self) -> Result<u32, Error>;

    /// Marks a datapoint as stale by inserting a new row with staled_at set to now
    async fn stale_datapoint(&self, params: &StaleDatapointParams) -> Result<(), Error>;

    /// Inserts a new datapoint into the dataset
    async fn insert_datapoint(&self, datapoint: &DatapointInsert) -> Result<(), Error>;

    /// Counts datapoints for a specific dataset and function
    async fn count_datapoints_for_dataset_function(
        &self,
        params: &CountDatapointsForDatasetFunctionParams,
    ) -> Result<u32, Error>;

    /// Gets the adjacent (previous and next) datapoint IDs for a given datapoint
    async fn get_adjacent_datapoint_ids(
        &self,
        params: &GetAdjacentDatapointIdsParams,
    ) -> Result<AdjacentDatapointIds, Error>;

    /// Gets a single datapoint by dataset name and ID
    async fn get_datapoint(&self, params: &GetDatapointParams) -> Result<Datapoint, Error>;

    /// Gets multiple datapoints by dataset name and IDs
    async fn get_datapoints(
        &self,
        dataset_name: &str,
        datapoint_ids: &[Uuid],
        allow_stale: bool,
    ) -> Result<Vec<Datapoint>, Error>;
}
