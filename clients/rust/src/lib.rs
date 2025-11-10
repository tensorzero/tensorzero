use std::{collections::HashMap, sync::Arc};

use tensorzero_core::client::DisplayOrDebug;
use tensorzero_core::db::inferences::InferenceQueries;
use tensorzero_core::db::HealthCheckable;
use tensorzero_core::endpoints::datasets::{InsertDatapointParams, StaleDatasetResponse};
use tensorzero_core::endpoints::optimization::{launch_optimization, launch_optimization_workflow};
use tensorzero_core::endpoints::stored_inferences::render_samples;
use tensorzero_core::endpoints::validate_tags;
use tensorzero_core::endpoints::workflow_evaluation_run::{
    WorkflowEvaluationRunEpisodeParams, WorkflowEvaluationRunEpisodeResponse,
};
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::stored_inference::StoredSample;
use uuid::Uuid;

// Re-export the core client from tensorzero-core

// Client core types
pub use tensorzero_core::client::{
    get_config_no_verify_credentials, Client, ClientBuilder, ClientBuilderMode, ClientMode,
    EmbeddedGateway, HTTPGateway,
};

// Client error types
pub use tensorzero_core::client::{
    err_to_http, with_embedded_timeout, ClientBuilderError, TensorZeroError,
    TensorZeroInternalError,
};

// Client input types
pub use tensorzero_core::client::{
    CacheParamsOptions, ClientInferenceParams, ClientInput, ClientInputMessage,
    ClientInputMessageContent, ClientSecretString,
};

// Input handling utilities
pub use tensorzero_core::client::input_handling;

// Re-export other commonly used types from tensorzero-core
pub use tensorzero_core::config::Config;
pub use tensorzero_core::db::clickhouse::query_builder::{
    BooleanMetricFilter, FloatComparisonOperator, FloatMetricFilter, InferenceFilter,
    TagComparisonOperator, TagFilter, TimeComparisonOperator, TimeFilter,
};
pub use tensorzero_core::db::datasets::{
    AdjacentDatapointIds, CountDatapointsForDatasetFunctionParams, DatapointInsert,
    DatasetDetailRow, DatasetQueries, DatasetQueryParams, GetAdjacentDatapointIdsParams,
    GetDatapointParams, GetDatapointsParams, GetDatasetMetadataParams, GetDatasetRowsParams,
    StaleDatapointParams,
};
pub use tensorzero_core::db::inferences::{InferenceOutputSource, ListInferencesParams};
pub use tensorzero_core::db::{ClickHouseConnection, ModelUsageTimePoint, TimeWindow};
pub use tensorzero_core::endpoints::datasets::v1::types::{
    CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsFromInferenceOutputSource,
    CreateDatapointsFromInferenceRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsRequest, CreateDatapointsResponse, CreateJsonDatapointRequest,
    DeleteDatapointsRequest, DeleteDatapointsResponse, GetDatapointsRequest, GetDatapointsResponse,
    JsonDatapointOutputUpdate, ListDatapointsRequest, UpdateChatDatapointRequest,
    UpdateDatapointMetadataRequest, UpdateDatapointRequest, UpdateDatapointsMetadataRequest,
    UpdateDatapointsRequest, UpdateDatapointsResponse, UpdateJsonDatapointRequest,
};
pub use tensorzero_core::endpoints::datasets::{
    ChatInferenceDatapoint, Datapoint, DatapointKind, JsonInferenceDatapoint,
    StoredChatInferenceDatapoint, StoredDatapoint,
};
pub use tensorzero_core::endpoints::feedback::FeedbackResponse;
pub use tensorzero_core::endpoints::feedback::Params as FeedbackParams;
pub use tensorzero_core::endpoints::inference::{
    InferenceOutput, InferenceParams, InferenceResponse, InferenceResponseChunk, InferenceStream,
};
pub use tensorzero_core::endpoints::object_storage::ObjectResponse;
pub use tensorzero_core::endpoints::optimization::{
    LaunchOptimizationParams, LaunchOptimizationWorkflowParams,
};
pub use tensorzero_core::endpoints::variant_probabilities::{
    GetVariantSamplingProbabilitiesParams, GetVariantSamplingProbabilitiesResponse,
};
pub use tensorzero_core::endpoints::workflow_evaluation_run::{
    WorkflowEvaluationRunParams, WorkflowEvaluationRunResponse,
};
pub use tensorzero_core::inference::types::storage::{StorageKind, StoragePath};
pub use tensorzero_core::inference::types::{
    Base64File, ContentBlockChunk, File, Input, InputMessage, InputMessageContent,
    ObjectStoragePointer, Role, System, Unknown, UnknownChunk, UrlFile,
};
pub use tensorzero_core::optimization::{OptimizationJobHandle, OptimizationJobInfo};
pub use tensorzero_core::stored_inference::{
    RenderedSample, StoredChatInference, StoredChatInferenceDatabase, StoredInference,
    StoredInferenceDatabase, StoredJsonInference,
};
pub use tensorzero_core::tool::{DynamicToolParams, Tool, ToolCallWrapper};
pub use tensorzero_core::utils::gateway::setup_clickhouse_without_config;

// Export quantile array from migration_0037
pub use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0037::QUANTILES;

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
//     page_size: Some(100),
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

    /// Gets datapoints by their IDs.
    ///
    /// # Arguments
    ///
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
    async fn create_from_inferences(
        &self,
        dataset_name: String,
        params: CreateDatapointsFromInferenceRequestParams,
        output_source: Option<CreateDatapointsFromInferenceOutputSource>,
    ) -> Result<CreateDatapointsResponse, TensorZeroError>;

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

    async fn experimental_list_inferences(
        &self,
        params: ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInference>, TensorZeroError>;

    // ================================================================
    // Optimization operations
    // ================================================================
    async fn experimental_render_samples<T: StoredSample + Send>(
        &self,
        stored_samples: Vec<T>,
        variants: HashMap<String, String>,
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
            client.parse_http_response(builder.send().await).await
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
                    tensorzero_core::endpoints::batch_inference::start_batch_inference(
                        gateway.handle.app_state.clone(),
                        params,
                        // We currently ban auth-enabled configs in embedded gateway mode,
                        // so we don't have an API key here
                        None,
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
                self.parse_http_response(builder.send().await).await
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
                self.parse_http_response(builder.send().await).await
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
        tracing::warn!("`Client::bulk_insert_datapoints` is deprecated. Use `Client::create_datapoints` instead.");
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
                self.parse_http_response(builder.send().await).await
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    let mut response = tensorzero_core::endpoints::datasets::v1::get_datapoints(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        &gateway.handle.app_state.config,
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
                self.parse_http_response(builder.send().await).await
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
                self.parse_http_response(builder.send().await).await
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
                self.parse_http_response(builder.send().await).await
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
        datapoint_ids: Vec<Uuid>,
    ) -> Result<GetDatapointsResponse, TensorZeroError> {
        let request = GetDatapointsRequest { ids: datapoint_ids };
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("v1/datasets/get_datapoints").map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/datasets/get_datapoints endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&request);
                self.parse_http_response(builder.send().await).await
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::get_datapoints(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        &gateway.handle.app_state.config,
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
                self.parse_http_response(builder.send().await).await
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    tensorzero_core::endpoints::datasets::v1::list_datapoints(
                        &gateway.handle.app_state.clickhouse_connection_info,
                        &gateway.handle.app_state.config,
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
                self.parse_http_response(builder.send().await).await
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
                self.parse_http_response(builder.send().await).await
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
                self.parse_http_response(builder.send().await).await
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

    async fn create_from_inferences(
        &self,
        dataset_name: String,
        params: CreateDatapointsFromInferenceRequestParams,
        output_source: Option<CreateDatapointsFromInferenceOutputSource>,
    ) -> Result<CreateDatapointsResponse, TensorZeroError> {
        let request = CreateDatapointsFromInferenceRequest {
            params,
            output_source,
        };
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join(&format!("v1/datasets/{dataset_name}/from_inferences")).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/datasets/{dataset_name}/from_inferences endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&request);
                self.parse_http_response(builder.send().await).await
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
            .map(|inf| inf.into_stored_inference(&gateway.handle.app_state.config))
            .collect();

        wire_inferences.map_err(|e| TensorZeroError::Other { source: e.into() })
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
                Ok(with_embedded_timeout(*timeout, async {
                    launch_optimization(
                        &gateway.handle.app_state.http_client,
                        params,
                        &gateway.handle.app_state.clickhouse_connection_info,
                        gateway.handle.app_state.config.clone(),
                    )
                    .await
                    .map_err(err_to_http)
                })
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
                with_embedded_timeout(*timeout, async {
                    launch_optimization_workflow(
                        &gateway.handle.app_state.http_client,
                        gateway.handle.app_state.config.clone(),
                        &gateway.handle.app_state.clickhouse_connection_info,
                        params,
                    )
                    .await
                    .map_err(err_to_http)
                })
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
                let resp = self.check_http_response(builder.send().await).await?;
                let encoded_handle = resp.text().await.map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error deserializing response: {}",
                            DisplayOrDebug {
                                val: e,
                                debug: self.verbose_errors,
                            }
                        ),
                    })
                    .into(),
                })?;
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
                    tensorzero_core::endpoints::optimization::poll_optimization(
                        &gateway.handle.app_state.http_client,
                        job_handle,
                        &gateway.handle.app_state.config.models.default_credentials,
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
                    self.parse_http_response(builder.send().await).await?;
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

    async fn get_variant_sampling_probabilities(
        &self,
        function_name: &str,
    ) -> Result<HashMap<String, f64>, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client
                    .base_url
                    .join("variant_sampling_probabilities")
                    .map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /variant_sampling_probabilities endpoint: {e}"
                            ),
                        })
                        .into(),
                    })?;
                let builder = client
                    .http_client
                    .get(url)
                    .query(&[("function_name", function_name)]);
                let response: GetVariantSamplingProbabilitiesResponse =
                    self.parse_http_response(builder.send().await).await?;
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
