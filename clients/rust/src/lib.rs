#![recursion_limit = "256"]

use std::{collections::HashMap, sync::Arc};
use tensorzero_core::config::snapshot::ConfigSnapshot;
use tensorzero_core::config::write_config_snapshot;
use tensorzero_core::db::HealthCheckable;
use tensorzero_core::db::inferences::InferenceQueries;
use tensorzero_core::endpoints::datasets::{InsertDatapointParams, StaleDatasetResponse};
use tensorzero_core::endpoints::stored_inferences::render_samples;
use tensorzero_core::endpoints::validate_tags;
use tensorzero_core::endpoints::workflow_evaluation_run::{
    WorkflowEvaluationRunEpisodeParams, WorkflowEvaluationRunEpisodeResponse,
};
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::stored_inference::StoredSample;
use tensorzero_optimizers::endpoints::{
    launch_optimization, launch_optimization_workflow, poll_optimization,
};
use uuid::Uuid;

// Re-export the core client from tensorzero-core

// Client core types
pub use tensorzero_core::client::{
    Client, ClientBuilder, ClientBuilderMode, ClientMode, EmbeddedGateway, HTTPGateway,
    PostgresConfig, get_config_no_verify_credentials,
};

// Client error types
pub use tensorzero_core::client::{
    ClientBuilderError, TensorZeroError, TensorZeroInternalError, err_to_http,
    with_embedded_timeout,
};

// Client input types
pub use tensorzero_core::client::{
    CacheParamsOptions, ClientInferenceParams, ClientSecretString, Input, InputMessage,
    InputMessageContent,
};

// Input handling utilities
pub use tensorzero_core::client::input_handling;

// Re-export other commonly used types from tensorzero-core
pub use tensorzero_core::config::Config;
pub use tensorzero_core::db::clickhouse::query_builder::{
    BooleanMetricFilter, FloatComparisonOperator, FloatMetricFilter, InferenceFilter, OrderBy,
    OrderByTerm, OrderDirection, TagComparisonOperator, TagFilter, TimeComparisonOperator,
    TimeFilter,
};
pub use tensorzero_core::db::datasets::{
    DatasetQueries, GetDatapointParams, GetDatapointsParams, GetDatasetMetadataParams,
};
pub use tensorzero_core::db::inferences::{InferenceOutputSource, ListInferencesParams};
pub use tensorzero_core::db::stored_datapoint::{
    StoredChatInferenceDatapoint, StoredDatapoint, StoredJsonInferenceDatapoint,
};
pub use tensorzero_core::db::{ClickHouseConnection, ModelUsageTimePoint, TimeWindow};
pub use tensorzero_core::endpoints::datasets::v1::types::{
    CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsFromInferenceRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsRequest, CreateDatapointsResponse,
    CreateJsonDatapointRequest, DatasetMetadata, DeleteDatapointsRequest, DeleteDatapointsResponse,
    GetDatapointsRequest, GetDatapointsResponse, JsonDatapointOutputUpdate, ListDatapointsRequest,
    ListDatasetsRequest, ListDatasetsResponse, UpdateChatDatapointRequest,
    UpdateDatapointMetadataRequest, UpdateDatapointRequest, UpdateDatapointsMetadataRequest,
    UpdateDatapointsRequest, UpdateDatapointsResponse, UpdateJsonDatapointRequest,
};
pub use tensorzero_core::endpoints::datasets::{
    ChatInferenceDatapoint, Datapoint, DatapointKind, JsonInferenceDatapoint,
};
pub use tensorzero_core::endpoints::feedback::FeedbackResponse;
pub use tensorzero_core::endpoints::feedback::Params as FeedbackParams;
pub use tensorzero_core::endpoints::inference::{
    ChatCompletionInferenceParams, InferenceOutput, InferenceParams, InferenceResponse,
    InferenceResponseChunk, InferenceStream,
};
pub use tensorzero_core::endpoints::internal::config::{
    GetConfigResponse, WriteConfigRequest, WriteConfigResponse,
};
pub use tensorzero_core::endpoints::object_storage::ObjectResponse;
pub use tensorzero_core::endpoints::stored_inferences::v1::types::{
    GetInferencesRequest, GetInferencesResponse, ListInferencesRequest,
};
pub use tensorzero_core::endpoints::variant_probabilities::{
    GetVariantSamplingProbabilitiesParams, GetVariantSamplingProbabilitiesResponse,
};
pub use tensorzero_core::endpoints::workflow_evaluation_run::{
    WorkflowEvaluationRunParams, WorkflowEvaluationRunResponse,
};
pub use tensorzero_core::inference::types::storage::{StorageKind, StoragePath};
pub use tensorzero_core::inference::types::{
    Base64File, ContentBlockChunk, File, ObjectStoragePointer, Role, System, Unknown, UnknownChunk,
    UrlFile, Usage,
};
pub use tensorzero_core::optimization::{OptimizationJobHandle, OptimizationJobInfo};
pub use tensorzero_core::stored_inference::{
    RenderedSample, StoredChatInference, StoredChatInferenceDatabase, StoredInference,
    StoredInferenceDatabase, StoredJsonInference,
};
pub use tensorzero_core::tool::{DynamicToolParams, FunctionTool, Tool, ToolCallWrapper};
pub use tensorzero_core::utils::gateway::setup_clickhouse_without_config;

// Export quantile array from migration_0037
pub use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0037::QUANTILES;

// Re-export optimization types from tensorzero-optimizers
pub use tensorzero_optimizers::endpoints::{
    LaunchOptimizationParams, LaunchOptimizationWorkflowParams,
};

// Keep git module for Git-related extension traits
mod git;

#[cfg(feature = "e2e_tests")]
pub mod test_helpers;

// Re-export observability for pyo3 feature
#[cfg(feature = "pyo3")]
pub use tensorzero_core::observability;

use crate::git::GitInfo;

// NOTE(shuyangli): For methods that delegate to APIs in the gateway, the arguments generally are flattened from the request type for
// ease of use, except when the type contains more than 2-3 fields or multiple fields with the same type (e.g. `ListDatapointsRequest`).
// This is because when reading the code outside of an IDE, it's often difficult to tell the arguments apart without argument names.
//
// To illustrate:
//
// It's easy to understand the semantics of methods that take few, unambiguous arguments:
// ```rust
// client.delete_datapoints("dataset-name", vec![uuid1, uuid2]);
// ```
//
// But it quickly gets confusing with more arguments or arguments with similar types:
// ```rust
// client.list_datapoints("dataset-name", None, Some(100), Some(0), None);
// ```
//
// In these cases, using the request type directly makes the code much more readable:
// ```rust
// client.list_datapoints("dataset-name", ListDatapointsRequest {
//     function_name: None,
//     limit: Some(100),
//     offset: Some(0),
//     filter: None,
// });
// ```

/// Extension trait for additional Client methods
#[async_trait::async_trait]
pub trait ClientExt {
    // ================================================================
    // Health checking
    // ================================================================
    async fn clickhouse_health(&self) -> Result<(), TensorZeroError>;

    // ================================================================
    // Dataset operations
    // ================================================================
    #[deprecated(since = "2025.11.3", note = "Use `create_datapoints` instead.")]
    async fn create_datapoints_legacy(
        &self,
        dataset_name: String,
        params: InsertDatapointParams,
    ) -> Result<Vec<Uuid>, TensorZeroError>;

    #[deprecated(since = "2025.11.3", note = "Use `create_datapoints` instead.")]
    async fn bulk_insert_datapoints(
        &self,
        dataset_name: String,
        params: InsertDatapointParams,
    ) -> Result<Vec<Uuid>, TensorZeroError>;

    #[deprecated(since = "2025.11.3", note = "Use `delete_datapoints` instead.")]
    async fn delete_datapoint(
        &self,
        dataset_name: String,
        datapoint_id: Uuid,
    ) -> Result<(), TensorZeroError>;

    #[deprecated(since = "2025.11.3", note = "Use `list_datapoints` instead.")]
    async fn list_datapoints_legacy(
        &self,
        dataset_name: String,
        function_name: Option<String>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Datapoint>, TensorZeroError>;

    #[deprecated(since = "2025.11.3", note = "Use `get_datapoints` instead.")]
    async fn get_datapoint(
        &self,
        dataset_name: String,
        datapoint_id: Uuid,
    ) -> Result<Datapoint, TensorZeroError>;

    #[deprecated(since = "2025.11.3", note = "Use `delete_dataset` instead")]
    async fn stale_dataset(
        &self,
        dataset_name: String,
    ) -> Result<StaleDatasetResponse, TensorZeroError>;

    /// Creates new datapoints in the dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - The name of the dataset to create the datapoints in.
    /// * `datapoints` - The datapoints to create.
    ///
    /// # Returns
    ///
    /// A `CreateDatapointsResponse` containing the IDs of the newly-created datapoints.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn create_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<CreateDatapointRequest>,
    ) -> Result<CreateDatapointsResponse, TensorZeroError>;

    /// Lists datapoints in the dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - The name of the dataset to list the datapoints from.
    /// * `request` - The request to list the datapoints.
    ///
    /// # Returns
    ///
    /// A `GetDatapointsResponse` containing the datapoints.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn list_datapoints(
        &self,
        dataset_name: String,
        request: ListDatapointsRequest,
    ) -> Result<GetDatapointsResponse, TensorZeroError>;

    /// Updates datapoints in the dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - The name of the dataset to update the datapoints in.
    /// * `datapoints` - The datapoints to update.
    ///
    /// # Returns
    ///
    /// A `UpdateDatapointsResponse` containing the IDs of the updated datapoints.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn update_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<UpdateDatapointRequest>,
    ) -> Result<UpdateDatapointsResponse, TensorZeroError>;

    /// Gets datapoints by their IDs and dataset name.
    /// Including the dataset name improves query performance because the dataset is part of the
    /// sorting key for datapoints.
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - The name of the dataset containing the datapoints.
    /// * `datapoint_ids` - The IDs of the datapoints to get.
    ///
    /// # Returns
    ///
    /// A `GetDatapointsResponse` containing the datapoints.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn get_datapoints(
        &self,
        dataset_name: Option<String>,
        datapoint_ids: Vec<Uuid>,
    ) -> Result<GetDatapointsResponse, TensorZeroError>;

    /// Updates the metadata of datapoints in the dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - The name of the dataset to update the metadata of.
    /// * `datapoints` - The datapoints to update the metadata of.
    ///
    /// # Returns
    ///
    /// A `UpdateDatapointsResponse` containing the IDs of the updated datapoints.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn update_datapoints_metadata(
        &self,
        dataset_name: String,
        datapoints: Vec<UpdateDatapointMetadataRequest>,
    ) -> Result<UpdateDatapointsResponse, TensorZeroError>;

    /// Deletes datapoints from the dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - The name of the dataset to delete the datapoints from.
    /// * `datapoint_ids` - The IDs of the datapoints to delete.
    ///
    /// # Returns
    ///
    /// A `DeleteDatapointsResponse` containing the number of deleted datapoints.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn delete_datapoints(
        &self,
        dataset_name: String,
        datapoint_ids: Vec<Uuid>,
    ) -> Result<DeleteDatapointsResponse, TensorZeroError>;

    /// Deletes a dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - The name of the dataset to delete.
    ///
    /// # Returns
    ///
    /// A `DeleteDatapointsResponse` containing the number of deleted datapoints.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn delete_dataset(
        &self,
        dataset_name: String,
    ) -> Result<DeleteDatapointsResponse, TensorZeroError>;

    /// Creates datapoints from inferences.
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - The name of the dataset to create the datapoints from.
    /// * `params` - The parameters for the creation.
    /// * `output_source` - The output source for the creation.
    ///
    /// # Returns
    ///
    /// A `CreateDatapointsResponse` containing the IDs of the newly-created datapoints.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn create_datapoints_from_inferences(
        &self,
        dataset_name: String,
        params: CreateDatapointsFromInferenceRequestParams,
    ) -> Result<CreateDatapointsResponse, TensorZeroError>;

    /// Lists all datasets with optional filtering and pagination.
    ///
    /// # Arguments
    ///
    /// * `request` - The request parameters for listing datasets.
    ///
    /// # Returns
    ///
    /// A `ListDatasetsResponse` containing metadata for matching datasets.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn list_datasets(
        &self,
        request: ListDatasetsRequest,
    ) -> Result<ListDatasetsResponse, TensorZeroError>;

    // ================================================================
    // Workflow evaluation operations
    // ================================================================
    async fn workflow_evaluation_run(
        &self,
        params: WorkflowEvaluationRunParams,
    ) -> Result<WorkflowEvaluationRunResponse, TensorZeroError>;

    async fn workflow_evaluation_run_episode(
        &self,
        run_id: Uuid,
        params: WorkflowEvaluationRunEpisodeParams,
    ) -> Result<WorkflowEvaluationRunEpisodeResponse, TensorZeroError>;

    // ================================================================
    // Inference operations
    // ================================================================
    #[cfg(feature = "e2e_tests")]
    async fn start_batch_inference(
        &self,
        params: tensorzero_core::endpoints::batch_inference::StartBatchInferenceParams,
    ) -> Result<
        tensorzero_core::endpoints::batch_inference::PrepareBatchInferenceOutput,
        TensorZeroError,
    >;

    #[deprecated(since = "2025.11.4", note = "Use `list_inferences` instead.")]
    async fn experimental_list_inferences(
        &self,
        params: ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInference>, TensorZeroError>;

    /// Gets specific inferences by their IDs.
    ///
    /// # Arguments
    ///
    /// * `inference_ids` - The IDs of the inferences to retrieve.
    /// * `function_name` - Optional function name to filter by (improves query performance).
    /// * `output_source` - Whether to return inference or demonstration output.
    ///
    /// # Returns
    ///
    /// A `GetInferencesResponse` containing the requested inferences.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn get_inferences(
        &self,
        inference_ids: Vec<Uuid>,
        function_name: Option<String>,
        output_source: InferenceOutputSource,
    ) -> Result<GetInferencesResponse, TensorZeroError>;

    /// Lists inferences with optional filtering, pagination, and sorting.
    ///
    /// # Arguments
    ///
    /// * `request` - The request parameters for listing inferences.
    ///
    /// # Returns
    ///
    /// A `GetInferencesResponse` containing the inferences that match the criteria.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn list_inferences(
        &self,
        request: ListInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroError>;

    // ================================================================
    // Optimization operations
    // ================================================================
    async fn experimental_render_samples<T: StoredSample + Send>(
        &self,
        stored_samples: Vec<T>,
        variants: HashMap<String, String>,
        concurrency: Option<usize>,
    ) -> Result<Vec<RenderedSample>, TensorZeroError>;

    async fn experimental_launch_optimization(
        &self,
        params: LaunchOptimizationParams,
    ) -> Result<OptimizationJobHandle, TensorZeroError>;

    async fn experimental_launch_optimization_workflow(
        &self,
        params: LaunchOptimizationWorkflowParams,
    ) -> Result<OptimizationJobHandle, TensorZeroError>;

    async fn experimental_poll_optimization(
        &self,
        handle: &OptimizationJobHandle,
    ) -> Result<OptimizationJobInfo, TensorZeroError>;

    // ================================================================
    // Variant sampling operations
    // ================================================================
    async fn get_variant_sampling_probabilities(
        &self,
        function_name: &str,
    ) -> Result<HashMap<String, f64>, TensorZeroError>;

    // ================================================================
    // Config access
    // ================================================================
    fn config(&self) -> Option<&Config>;

    fn get_config(&self) -> Result<Arc<Config>, TensorZeroError>;

    /// Gets a config snapshot by hash, or the live config if no hash is provided.
    ///
    /// # Arguments
    ///
    /// * `hash` - Optional hash of the config snapshot to retrieve. If `None`, returns the live config.
    ///
    /// # Returns
    ///
    /// A `GetConfigResponse` containing the config snapshot.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails or the config snapshot is not found.
    async fn get_config_snapshot(
        &self,
        hash: Option<&str>,
    ) -> Result<GetConfigResponse, TensorZeroError>;

    /// Writes a config snapshot to the database.
    ///
    /// If a config with the same hash already exists, tags are merged
    /// (new tags override existing keys) and `created_at` is preserved.
    ///
    /// # Arguments
    ///
    /// * `request` - The config to write, including optional extra_templates and tags.
    ///
    /// # Returns
    ///
    /// A `WriteConfigResponse` containing the computed hash of the config.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn write_config(
        &self,
        request: WriteConfigRequest,
    ) -> Result<WriteConfigResponse, TensorZeroError>;

    #[cfg(any(feature = "e2e_tests", feature = "pyo3"))]
    fn get_app_state_data(&self) -> Option<&tensorzero_core::utils::gateway::AppStateData>;
}

// Private helper for creating datapoints (shared by create_datapoints and bulk_insert_datapoints)
async fn create_datapoints_internal(
    client: &Client,
    dataset_name: String,
    params: InsertDatapointParams,
    endpoint_path: &str,
) -> Result<Vec<Uuid>, TensorZeroError> {
    match client.mode() {
        ClientMode::HTTPGateway(http_client) => {
            let url = http_client.base_url.join(&format!("datasets/{dataset_name}/{endpoint_path}")).map_err(|e| TensorZeroError::Other {
                source: Error::new(ErrorDetails::InvalidBaseUrl {
                    message: format!("Failed to join base URL with /datasets/{dataset_name}/{endpoint_path} endpoint: {e}"),
                })
                .into(),
            })?;
            let builder = http_client.http_client.post(url).json(&params);
            Ok(http_client.send_and_parse_http_response(builder).await?.0)
        }
        ClientMode::EmbeddedGateway { gateway, timeout } => {
            Ok(with_embedded_timeout(*timeout, async {
                tensorzero_core::endpoints::datasets::insert_datapoint(
                    dataset_name,
                    params,
                    &gateway.handle.app_state.config,
                    &gateway.handle.app_state.http_client,
                    &gateway.handle.app_state.clickhouse_connection_info,
                )
                .await
                .map_err(err_to_http)
            })
            .await?)
        }
    }
}

#[async_trait::async_trait]
impl ClientExt for Client {
    /// Queries the health of the ClickHouse database
    /// This does nothing in `ClientMode::HTTPGateway`
    async fn clickhouse_health(&self) -> Result<(), TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(_) => Ok(()),
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => gateway
                .handle
                .app_state
                .clickhouse_connection_info
                .health()
                .await
                .map_err(|e| TensorZeroError::Other { source: e.into() }),
        }
    }

    /// Gets the config from the embedded gateway
    /// Returns None for HTTP gateway mode
    fn config(&self) -> Option<&Config> {
        match self.mode() {
            ClientMode::HTTPGateway(_) => None,
            ClientMode::EmbeddedGateway { gateway, .. } => Some(&gateway.handle.app_state.config),
        }
    }

    #[cfg(feature = "e2e_tests")]
    async fn start_batch_inference(
        &self,
        params: tensorzero_core::endpoints::batch_inference::StartBatchInferenceParams,
    ) -> Result<
        tensorzero_core::endpoints::batch_inference::PrepareBatchInferenceOutput,
        TensorZeroError,
    > {
        match self.mode() {
            ClientMode::HTTPGateway(_) => Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InternalError {
                    message: "batch_inference is not yet implemented for HTTPGateway mode"
                        .to_string(),
                })
                .into(),
            }),
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    Box::pin(
                        tensorzero_core::endpoints::batch_inference::start_batch_inference(
                            gateway.handle.app_state.clone(),
                            params,
                            // We currently ban auth-enabled configs in embedded gateway mode,
                            // so we don't have an API key here
                            None,
                        ),
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await?)
            }
        }
    }

    async fn workflow_evaluation_run(
        &self,
        mut params: WorkflowEvaluationRunParams,
    ) -> Result<WorkflowEvaluationRunResponse, TensorZeroError> {
        // We validate the tags here since we're going to add git information to the tags afterwards and set internal to true
        validate_tags(&params.tags, false)
            .map_err(|e| TensorZeroError::Other { source: e.into() })?;

        // Apply the git information to the tags so it gets stored for our workflow evaluation run
        if let Ok(git_info) = GitInfo::new() {
            params.tags.extend(git_info.into_tags());
        }

        // Set internal to true so we don't validate the tags again
        params.internal = true;

        // Automatically add internal tag when internal=true
        params
            .tags
            .insert("tensorzero::internal".to_string(), "true".to_string());

        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("workflow_evaluation_run").map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /workflow_evaluation_run endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&params);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::workflow_evaluation_run::workflow_evaluation_run(
                        gateway.handle.app_state.clone(),
                        params,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await?)
            }
        }
    }

    async fn workflow_evaluation_run_episode(
        &self,
        run_id: Uuid,
        params: WorkflowEvaluationRunEpisodeParams,
    ) -> Result<WorkflowEvaluationRunEpisodeResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("workflow_evaluation_run/{run_id}/episode")).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /workflow_evaluation_run/{run_id}/episode endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&params);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::workflow_evaluation_run::workflow_evaluation_run_episode(
                        gateway.handle.app_state.clone(),
                        run_id,
                        params,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await?)
            }
        }
    }

    /// DEPRECATED: Use `create_datapoints` instead.
    async fn create_datapoints_legacy(
        &self,
        dataset_name: String,
        params: InsertDatapointParams,
    ) -> Result<Vec<Uuid>, TensorZeroError> {
        // No warning because the python client still uses it.
        create_datapoints_internal(self, dataset_name, params, "datapoints").await
    }

    /// DEPRECATED: Use `create_datapoints` instead.
    async fn bulk_insert_datapoints(
        &self,
        dataset_name: String,
        params: InsertDatapointParams,
    ) -> Result<Vec<Uuid>, TensorZeroError> {
        tracing::warn!(
            "`Client::bulk_insert_datapoints` is deprecated. Use `Client::create_datapoints` instead."
        );
        create_datapoints_internal(self, dataset_name, params, "datapoints/bulk").await
    }

    /// DEPRECATED: Use `list_datapoints` instead.
    async fn list_datapoints_legacy(
        &self,
        dataset_name: String,
        function_name: Option<String>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Datapoint>, TensorZeroError> {
        let response = self
            .list_datapoints(
                dataset_name,
                ListDatapointsRequest {
                    function_name,
                    limit,
                    offset,
                    ..Default::default()
                },
            )
            .await?;
        Ok(response.datapoints)
    }

    /// DEPRECATED: Use `delete_datapoints` instead.
    async fn delete_datapoint(
        &self,
        dataset_name: String,
        datapoint_id: Uuid,
    ) -> Result<(), TensorZeroError> {
        let response = self
            .delete_datapoints(dataset_name, vec![datapoint_id])
            .await?;
        if response.num_deleted_datapoints == 0 {
            return Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Datapoint with ID {datapoint_id} not found"),
                })
                .into(),
            });
        }
        Ok(())
    }

    /// DEPRECATED: Use `get_datapoints` instead.
    async fn get_datapoint(
        &self,
        dataset_name: String,
        datapoint_id: Uuid,
    ) -> Result<Datapoint, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("datasets/{dataset_name}/datapoints/{datapoint_id}")).map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!("Failed to join base URL with /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint: {e}"),
                        })
                        .into(),
                    })?;
                let builder = client.http_client.get(url);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    let mut response = tensorzero_core::endpoints::datasets::v1::get_datapoints(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        Some(dataset_name.clone()),
                        GetDatapointsRequest {
                            ids: vec![datapoint_id],
                        },
                    )
                    .await
                    .map_err(err_to_http)?;

                    if response.datapoints.is_empty() {
                        // We explicitly construct an HTTP error here because python client expects it.
                        return Err(err_to_http(Error::new(ErrorDetails::DatapointNotFound {
                            dataset_name,
                            datapoint_id,
                        })));
                    }
                    Ok(response.datapoints.swap_remove(0))
                })
                .await
            }
        }
    }

    /// DEPRECATED: Use `delete_dataset` instead.
    /// Stales all datapoints in a dataset that have not been staled yet.
    /// This is a soft deletion, so evaluation runs will still refer to it.
    /// Returns the number of datapoints that were staled as {num_staled_datapoints: u64}.
    async fn stale_dataset(
        &self,
        dataset_name: String,
    ) -> Result<StaleDatasetResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::stale_dataset(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        &dataset_name,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("datasets/{dataset_name}")).map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!("Failed to join base URL with /datasets/{dataset_name} endpoint: {e}"),
                        })
                        .into(),
                    })?;
                let builder = client.http_client.delete(url);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
        }
    }

    async fn create_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<CreateDatapointRequest>,
    ) -> Result<CreateDatapointsResponse, TensorZeroError> {
        let request = CreateDatapointsRequest { datapoints };
        match self.mode() {
            ClientMode::HTTPGateway(http_client) => {
                let url = http_client.base_url.join(&format!("v1/datasets/{dataset_name}/datapoints")).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/datasets/{dataset_name}/datapoints endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = http_client.http_client.post(url).json(&request);
                Ok(http_client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::create_datapoints(
                        &gateway.handle.app_state.config,
                        &gateway.handle.app_state.http_client,
                        &gateway.handle.app_state.clickhouse_connection_info,
                        &dataset_name,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn update_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<UpdateDatapointRequest>,
    ) -> Result<UpdateDatapointsResponse, TensorZeroError> {
        let request = UpdateDatapointsRequest { datapoints };
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("v1/datasets/{dataset_name}/datapoints")).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/datasets/{dataset_name}/datapoints endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.patch(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::update_datapoints(
                        &gateway.handle.app_state,
                        &dataset_name,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn get_datapoints(
        &self,
        dataset_name: Option<String>,
        datapoint_ids: Vec<Uuid>,
    ) -> Result<GetDatapointsResponse, TensorZeroError> {
        let request = GetDatapointsRequest { ids: datapoint_ids };
        match self.mode() {
            ClientMode::HTTPGateway(http_client) => {
                let url = match dataset_name.as_ref() {
                    Some(dataset_name) => http_client
                        .base_url
                        .join(&format!("v1/datasets/{dataset_name}/get_datapoints"))
                        .map_err(|e| TensorZeroError::Other {
                            source: Error::new(ErrorDetails::InvalidBaseUrl {
                                message: format!(
                                    "Failed to join base URL with /v1/datasets/{dataset_name}/get_datapoints endpoint: {e}"
                                ),
                            })
                            .into(),
                        })?,
                    None => http_client.base_url.join("v1/datasets/get_datapoints").map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!("Failed to join base URL with /v1/datasets/get_datapoints endpoint: {e}"),
                        })
                        .into(),
                    })?,
                };
                let builder = http_client.http_client.post(url).json(&request);
                Ok(http_client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::get_datapoints(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        dataset_name,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn list_datapoints(
        &self,
        dataset_name: String,
        request: ListDatapointsRequest,
    ) -> Result<GetDatapointsResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("v1/datasets/{dataset_name}/list_datapoints")).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/datasets/{dataset_name}/list_datapoints endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::list_datapoints(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        dataset_name,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn update_datapoints_metadata(
        &self,
        dataset_name: String,
        datapoints: Vec<UpdateDatapointMetadataRequest>,
    ) -> Result<UpdateDatapointsResponse, TensorZeroError> {
        let request = UpdateDatapointsMetadataRequest { datapoints };
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("v1/datasets/{dataset_name}/datapoints/metadata")).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/datasets/{dataset_name}/datapoints/metadata endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.patch(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::update_datapoints_metadata(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        &dataset_name,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn delete_datapoints(
        &self,
        dataset_name: String,
        datapoint_ids: Vec<Uuid>,
    ) -> Result<DeleteDatapointsResponse, TensorZeroError> {
        let request = DeleteDatapointsRequest { ids: datapoint_ids };
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("v1/datasets/{dataset_name}/datapoints")).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/datasets/{dataset_name}/datapoints endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.delete(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::delete_datapoints(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        &dataset_name,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn delete_dataset(
        &self,
        dataset_name: String,
    ) -> Result<DeleteDatapointsResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("v1/datasets/{dataset_name}")).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/datasets/{dataset_name} endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.delete(url);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::delete_dataset(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        &dataset_name,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn create_datapoints_from_inferences(
        &self,
        dataset_name: String,
        params: CreateDatapointsFromInferenceRequestParams,
    ) -> Result<CreateDatapointsResponse, TensorZeroError> {
        let request = CreateDatapointsFromInferenceRequest { params };
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("v1/datasets/{dataset_name}/from_inferences")).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/datasets/{dataset_name}/from_inferences endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::create_from_inferences(
                        &gateway.handle.app_state.config,
                        &gateway.handle.app_state.clickhouse_connection_info,
                        dataset_name,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await?)
            }
        }
    }

    async fn list_datasets(
        &self,
        request: ListDatasetsRequest,
    ) -> Result<ListDatasetsResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let ListDatasetsRequest {
                    function_name,
                    limit,
                    offset,
                } = &request;
                let mut url = client.base_url.join("internal/datasets").map_err(|e| {
                    TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /internal/datasets endpoint: {e}"
                            ),
                        })
                        .into(),
                    }
                })?;
                // Add query params
                if let Some(function_name) = function_name {
                    url.query_pairs_mut()
                        .append_pair("function_name", function_name);
                }
                if let Some(limit) = limit {
                    url.query_pairs_mut()
                        .append_pair("limit", &limit.to_string());
                }
                if let Some(offset) = offset {
                    url.query_pairs_mut()
                        .append_pair("offset", &offset.to_string());
                }
                let builder = client.http_client.get(url);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::list_datasets(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    /// Query the Clickhouse database for inferences.
    ///
    /// This function is only available in EmbeddedGateway mode.
    ///
    /// # Arguments
    ///
    /// * `function_name` - The name of the function to query.
    /// * `variant_name` - The name of the variant to query. Optional
    /// * `filters` - A filter tree to apply to the query. Optional
    /// * `output_source` - The source of the output to query. "inference" or "demonstration"
    /// * `limit` - The maximum number of inferences to return. Optional
    /// * `offset` - The offset to start from. Optional
    /// * `format` - The format to return the inferences in. For now, only "JSONEachRow" is supported.
    async fn experimental_list_inferences(
        &self,
        params: ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInference>, TensorZeroError> {
        // TODO: consider adding a flag that returns the generated sql query
        let ClientMode::EmbeddedGateway { gateway, .. } = self.mode() else {
            return Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InvalidClientMode {
                    mode: "Http".to_string(),
                    message: "This function is only available in EmbeddedGateway mode".to_string(),
                })
                .into(),
            });
        };
        let inferences = gateway
            .handle
            .app_state
            .clickhouse_connection_info
            .list_inferences(&gateway.handle.app_state.config, &params)
            .await
            .map_err(err_to_http)?;

        // Convert storage types to wire types
        let wire_inferences: Result<Vec<StoredInference>, _> = inferences
            .into_iter()
            .map(StoredInferenceDatabase::into_stored_inference)
            .collect();

        wire_inferences.map_err(|e| TensorZeroError::Other { source: e.into() })
    }

    async fn get_inferences(
        &self,
        inference_ids: Vec<Uuid>,
        function_name: Option<String>,
        output_source: InferenceOutputSource,
    ) -> Result<GetInferencesResponse, TensorZeroError> {
        let request = GetInferencesRequest {
            ids: inference_ids,
            function_name,
            output_source,
        };
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("v1/inferences/get_inferences").map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/inferences/get_inferences endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::stored_inferences::v1::get_inferences(
                        &gateway.handle.app_state.config,
                        &gateway.handle.app_state.clickhouse_connection_info,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn list_inferences(
        &self,
        request: ListInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("v1/inferences/list_inferences").map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/inferences/list_inferences endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::stored_inferences::v1::list_inferences(
                        &gateway.handle.app_state.config,
                        &gateway.handle.app_state.clickhouse_connection_info,
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    /// There are two things that need to happen in this function:
    /// 1. We need to resolve all network resources (e.g. images) in the inference examples.
    /// 2. We need to prepare all messages into "simple" messages that have been templated for a particular variant.
    ///    To do this, we need to know what variant to use for each function that might appear in the data.
    ///
    /// IMPORTANT: For now, this function drops datapoints which are bad, e.g. ones where templating fails, the function
    ///            has no variant specified, or where the process of downloading resources fails.
    ///            In future we will make this behavior configurable by the caller.
    async fn experimental_render_samples<T: StoredSample + Send>(
        &self,
        stored_samples: Vec<T>,
        variants: HashMap<String, String>, // Map from function name to variant name
        concurrency: Option<usize>,
    ) -> Result<Vec<RenderedSample>, TensorZeroError> {
        let ClientMode::EmbeddedGateway { gateway, .. } = self.mode() else {
            return Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InvalidClientMode {
                    mode: "Http".to_string(),
                    message: "This function is only available in EmbeddedGateway mode".to_string(),
                })
                .into(),
            });
        };
        render_samples(
            gateway.handle.app_state.config.clone(),
            stored_samples,
            variants,
            concurrency,
        )
        .await
        .map_err(err_to_http)
    }

    /// Launch an optimization job.
    async fn experimental_launch_optimization(
        &self,
        params: LaunchOptimizationParams,
    ) -> Result<OptimizationJobHandle, TensorZeroError> {
        match self.mode() {
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(Box::pin(with_embedded_timeout(*timeout, async {
                    launch_optimization(
                        &gateway.handle.app_state.http_client,
                        params,
                        &gateway.handle.app_state.clickhouse_connection_info,
                        gateway.handle.app_state.config.clone(),
                    )
                    .await
                    .map_err(err_to_http)
                }))
                .await?)
            }
            ClientMode::HTTPGateway(_) => Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InvalidClientMode {
                    mode: "Http".to_string(),
                    message: "This function is only available in EmbeddedGateway mode".to_string(),
                })
                .into(),
            }),
        }
    }

    /// Start an optimization job.
    /// NOTE: This is the composition of `list_inferences`, `render_inferences`, and `launch_optimization`.
    async fn experimental_launch_optimization_workflow(
        &self,
        params: LaunchOptimizationWorkflowParams,
    ) -> Result<OptimizationJobHandle, TensorZeroError> {
        match self.mode() {
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Box::pin(with_embedded_timeout(*timeout, async {
                    launch_optimization_workflow(
                        &gateway.handle.app_state.http_client,
                        gateway.handle.app_state.config.clone(),
                        &gateway.handle.app_state.clickhouse_connection_info,
                        params,
                    )
                    .await
                    .map_err(err_to_http)
                }))
                .await
            }
            ClientMode::HTTPGateway(client) => {
                let url = client
                    .base_url
                    .join("experimental_optimization_workflow")
                    .map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /optimization_workflow endpoint: {e}"
                            ),
                        })
                        .into(),
                    })?;
                let builder = client.http_client.post(url).json(&params);
                let encoded_handle = client.send_request(builder).await?;
                let job_handle = OptimizationJobHandle::from_base64_urlencoded(&encoded_handle)
                    .map_err(|e| TensorZeroError::Other { source: e.into() })?;
                Ok(job_handle)
            }
        }
    }

    /// Poll an optimization job for status.
    async fn experimental_poll_optimization(
        &self,
        job_handle: &OptimizationJobHandle,
    ) -> Result<OptimizationJobInfo, TensorZeroError> {
        match self.mode() {
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    poll_optimization(
                        &gateway.handle.app_state.http_client,
                        job_handle,
                        &gateway.handle.app_state.config.models.default_credentials,
                        &gateway.handle.app_state.config.provider_types,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await?)
            }
            ClientMode::HTTPGateway(client) => {
                let encoded_job_handle = job_handle
                    .to_base64_urlencoded()
                    .map_err(|e| TensorZeroError::Other { source: e.into() })?;
                let url = client
                    .base_url
                    .join(&format!("experimental_optimization/{encoded_job_handle}"))
                    .map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!("Failed to join base URL with /optimization/{encoded_job_handle} endpoint: {e}"),
                        })
                        .into(),
                    })?;
                let builder = client.http_client.get(url);
                let resp: OptimizationJobInfo =
                    client.send_and_parse_http_response(builder).await?.0;
                Ok(resp)
            }
        }
    }

    fn get_config(&self) -> Result<Arc<Config>, TensorZeroError> {
        match self.mode() {
            ClientMode::EmbeddedGateway { gateway, .. } => {
                Ok(gateway.handle.app_state.config.clone())
            }
            ClientMode::HTTPGateway(_) => Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InvalidClientMode {
                    mode: "Http".to_string(),
                    message: "This function is only available in EmbeddedGateway mode".to_string(),
                })
                .into(),
            }),
        }
    }

    async fn get_config_snapshot(
        &self,
        hash: Option<&str>,
    ) -> Result<GetConfigResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let endpoint = match hash {
                    Some(h) => format!("internal/config/{h}"),
                    None => "internal/config".to_string(),
                };
                let url = client
                    .base_url
                    .join(&endpoint)
                    .map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /{endpoint} endpoint: {e}"
                            ),
                        })
                        .into(),
                    })?;
                let builder = client.http_client.get(url);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    use tensorzero_core::db::ConfigQueries;
                    let snapshot_hash = match hash {
                        Some(h) => h.parse().map_err(|_| {
                            err_to_http(Error::new(ErrorDetails::ConfigSnapshotNotFound {
                                snapshot_hash: h.to_string(),
                            }))
                        })?,
                        None => gateway.handle.app_state.config.hash.clone(),
                    };
                    let snapshot = gateway
                        .handle
                        .app_state
                        .clickhouse_connection_info
                        .get_config_snapshot(snapshot_hash)
                        .await
                        .map_err(err_to_http)?;
                    Ok(GetConfigResponse {
                        hash: snapshot.hash.to_string(),
                        config: snapshot.config.try_into().map_err(|e: &'static str| {
                            err_to_http(Error::new(ErrorDetails::Config {
                                message: e.to_string(),
                            }))
                        })?,
                        extra_templates: snapshot.extra_templates,
                        tags: snapshot.tags,
                    })
                })
                .await
            }
        }
    }

    async fn write_config(
        &self,
        request: WriteConfigRequest,
    ) -> Result<WriteConfigResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("internal/config").map_err(|e| {
                    TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /internal/config endpoint: {e}"
                            ),
                        })
                        .into(),
                    }
                })?;
                let builder = client.http_client.post(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Box::pin(with_embedded_timeout(*timeout, async {
                    let mut snapshot = ConfigSnapshot::new(request.config, request.extra_templates)
                        .map_err(err_to_http)?;
                    snapshot.tags = request.tags;

                    let hash = snapshot.hash.to_string();

                    write_config_snapshot(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        snapshot,
                    )
                    .await
                    .map_err(err_to_http)?;

                    Ok(WriteConfigResponse { hash })
                }))
                .await
            }
        }
    }

    async fn get_variant_sampling_probabilities(
        &self,
        function_name: &str,
    ) -> Result<HashMap<String, f64>, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let endpoint = format!("internal/functions/{function_name}/variant_sampling_probabilities");
                let url = client
                    .base_url
                    .join(&endpoint)
                    .map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /internal/functions/{function_name}/variant_sampling_probabilities endpoint: {e}"
                            ),
                        })
                        .into(),
                    })?;
                let builder = client.http_client.get(url);
                let response: GetVariantSamplingProbabilitiesResponse =
                    client.send_and_parse_http_response(builder).await?.0;
                Ok(response.probabilities)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    let response = tensorzero_core::endpoints::variant_probabilities::get_variant_sampling_probabilities(
                        gateway.handle.app_state.clone(),
                        GetVariantSamplingProbabilitiesParams {
                            function_name: function_name.to_string(),
                        },
                    )
                    .await
                    .map_err(err_to_http)?;
                    Ok(response.probabilities)
                })
                .await?)
            }
        }
    }

    #[cfg(any(feature = "e2e_tests", feature = "pyo3"))]
    fn get_app_state_data(&self) -> Option<&tensorzero_core::utils::gateway::AppStateData> {
        match self.mode() {
            ClientMode::EmbeddedGateway { gateway, .. } => Some(&gateway.handle.app_state),
            ClientMode::HTTPGateway(_) => None,
        }
    }
}
