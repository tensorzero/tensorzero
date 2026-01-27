use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

use crate::config::{MetricConfigLevel, MetricConfigType};
use crate::db::clickhouse::query_builder::{DatapointFilter, FloatComparisonOperator};
use crate::db::stored_datapoint::StoredDatapoint;
use crate::endpoints::datasets::v1::types::DatapointOrderBy;
use crate::error::Error;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MetricFilter {
    pub metric: String,
    pub metric_type: MetricConfigType,
    pub operator: FloatComparisonOperator,
    pub threshold: f64,
    pub join_on: MetricConfigLevel,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum DatasetOutputSource {
    // When generating a dataset, don't include any output.
    None,
    // When generating a dataset, include original inference output.
    Inference,
    // When generating a dataset, include any linked demonstration as output.
    Demonstration,
}

/// Parameters to query for dataset metadata (by aggregating over the datapoint tables).
#[derive(Deserialize)]
pub struct GetDatasetMetadataParams {
    /// Only select datasets matching a specific function.
    pub function_name: Option<String>,

    /// The maximum number of datasets to return.
    pub limit: Option<u32>,

    /// The number of datasets to skip before starting to return results.
    pub offset: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetMetadata {
    pub dataset_name: String,
    pub count: u32,
    pub last_updated: String,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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
    /// Gets dataset metadata (name, count, last updated)
    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error>;

    /// Inserts a batch of datapoints into the database
    /// Internally separates chat and JSON datapoints and writes them to the appropriate tables
    /// Returns the number of rows written.
    async fn insert_datapoints(&self, datapoints: &[StoredDatapoint]) -> Result<u64, Error>;

    /// Counts datapoints for a dataset, optionally filtered by function name.
    async fn count_datapoints_for_dataset(
        &self,
        dataset_name: &str,
        function_name: Option<&str>,
    ) -> Result<u64, Error>;

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

    /// Clones datapoints to a target dataset, preserving all fields except id and dataset_name.
    ///
    /// For each source datapoint ID, generates a new UUID and attempts to clone the datapoint
    /// to the target dataset. The operation handles both Chat and Json datapoints.
    ///
    /// Returns a Vec with the same length as `source_datapoint_ids`, where each element is:
    /// - `Some(new_id)` if the source datapoint was found and cloned successfully
    /// - `None` if the source datapoint doesn't exist
    async fn clone_datapoints(
        &self,
        target_dataset_name: &str,
        source_datapoint_ids: &[Uuid],
    ) -> Result<Vec<Option<Uuid>>, Error>;
}
