use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use tensorzero_core::error::Error;
use tracing::instrument;

use tensorzero_core::endpoints::feedback::{FeedbackToolParams, feedback};
use tensorzero_core::endpoints::inference::{InferenceOutput, InferenceToolParams, inference};

use tensorzero_core::endpoints::datasets::v1::types::{
    CreateDatapointsFromInferenceRequest, CreateDatapointsFromInferencesToolParams,
    CreateDatapointsRequest, CreateDatapointsToolParams, DeleteDatapointsRequest,
    DeleteDatapointsToolParams, GetDatapointsRequest, GetDatapointsToolParams,
    ListDatapointsToolParams, ListDatasetsRequest, UpdateDatapointsRequest,
    UpdateDatapointsToolParams,
};
use tensorzero_core::endpoints::datasets::v1::{
    create_datapoints, create_from_inferences, delete_datapoints, get_datapoints, list_datapoints,
    list_datasets, update_datapoints,
};
use tensorzero_core::endpoints::episodes::internal::{
    ListEpisodesRequest, ListEpisodesResponse, list_episodes,
};
use tensorzero_core::endpoints::feedback::internal::{
    GetFeedbackByTargetIdToolParams, GetFeedbackByVariantToolParams,
    GetLatestFeedbackByMetricToolParams, get_feedback_by_target_id, get_feedback_by_variant,
    get_latest_feedback_id_by_metric,
};
use tensorzero_core::endpoints::stored_inferences::v1::types::{
    GetInferencesRequest, ListInferencesRequest,
};
use tensorzero_core::endpoints::stored_inferences::v1::{get_inferences, list_inferences};
use tensorzero_core::utils::gateway::AppStateData;

/// Converts a TensorZero error into either a tool-level error result (for client errors)
/// or an MCP protocol error (for server errors).
fn handle_tool_error(e: Error) -> Result<CallToolResult, McpError> {
    if e.status_code().is_client_error() {
        Ok(CallToolResult::error(vec![Content::text(e.to_string())]))
    } else {
        Err(McpError::internal_error(e.to_string(), None))
    }
}

#[derive(Clone)]
pub(crate) struct TensorZeroMcpServer {
    app_state: Arc<AppStateData>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl TensorZeroMcpServer {
    pub fn new(app_state: Arc<AppStateData>) -> Self {
        Self {
            app_state,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "List inferences stored in TensorZero with filtering, pagination, and sorting. Returns inference data including inputs, outputs, function/variant names, timestamps, and tags."
    )]
    #[instrument(skip_all)]
    async fn list_inferences(
        &self,
        Parameters(request): Parameters<ListInferencesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let response = match list_inferences(&self.app_state.config, &database, request).await {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Retrieve specific inferences by their IDs. Returns full inference data including inputs, outputs, function/variant names, timestamps, and tags."
    )]
    #[instrument(skip_all)]
    async fn get_inferences(
        &self,
        Parameters(request): Parameters<GetInferencesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let response = match get_inferences(&self.app_state.config, &database, request).await {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "List available datasets with optional filtering by function name and pagination. Returns dataset names, datapoint counts, and last updated timestamps."
    )]
    #[instrument(skip_all)]
    async fn list_datasets(
        &self,
        Parameters(request): Parameters<ListDatasetsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let response = match list_datasets(&database, request).await {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "List datapoints in a dataset with optional filtering and pagination. Can filter by function name, tags, time ranges, and order results."
    )]
    #[instrument(skip_all)]
    async fn list_datapoints(
        &self,
        Parameters(params): Parameters<ListDatapointsToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let response = match list_datapoints(&database, params.dataset_name, params.request).await {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Retrieve specific datapoints by their IDs. Optionally provide a dataset name for better query performance."
    )]
    #[instrument(skip_all)]
    async fn get_datapoints(
        &self,
        Parameters(params): Parameters<GetDatapointsToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let request = GetDatapointsRequest { ids: params.ids };
        let response = match get_datapoints(&database, params.dataset_name, request).await {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "List episodes with pagination and optional filtering by function name. Returns episode IDs, inference counts, time ranges, and last inference IDs."
    )]
    #[instrument(skip_all)]
    async fn list_episodes(
        &self,
        Parameters(request): Parameters<ListEpisodesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let episodes = match list_episodes(
            &database,
            &self.app_state.config,
            request.limit,
            request.before,
            request.after,
            request.function_name,
            request.filters,
        )
        .await
        {
            Ok(episodes) => episodes,
            Err(e) => return handle_tool_error(e),
        };

        let response = ListEpisodesResponse { episodes };
        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Get all feedback for a given target (inference or episode). Returns boolean metrics, float metrics, comments, and demonstrations."
    )]
    #[instrument(skip_all)]
    async fn get_feedback_by_target_id(
        &self,
        Parameters(params): Parameters<GetFeedbackByTargetIdToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let response =
            match get_feedback_by_target_id(&database, params.target_id, None, None, params.limit)
                .await
            {
                Ok(response) => response,
                Err(e) => return handle_tool_error(e),
            };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Get the latest feedback ID for each metric for a given target (inference). Returns a map from metric name to the latest feedback ID."
    )]
    #[instrument(skip_all)]
    async fn get_latest_feedback_by_metric(
        &self,
        Parameters(params): Parameters<GetLatestFeedbackByMetricToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let response = match get_latest_feedback_id_by_metric(&database, params.target_id).await {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Create new datapoints in a dataset. Supports both chat and JSON datapoint types with input, output, tags, and metadata."
    )]
    #[instrument(skip_all)]
    async fn create_datapoints(
        &self,
        Parameters(params): Parameters<CreateDatapointsToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let request = CreateDatapointsRequest {
            datapoints: params.datapoints,
        };
        let response = match create_datapoints(
            &self.app_state.config,
            &self.app_state.http_client,
            &database,
            &params.dataset_name,
            request,
        )
        .await
        {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Update existing datapoints in a dataset. Creates new versions of datapoints with updated input, output, tags, or metadata."
    )]
    #[instrument(skip_all)]
    async fn update_datapoints(
        &self,
        Parameters(params): Parameters<UpdateDatapointsToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let request = UpdateDatapointsRequest {
            datapoints: params.datapoints,
        };
        let response = match update_datapoints(&self.app_state, &params.dataset_name, request).await
        {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Delete datapoints from a dataset by their IDs. This is a soft delete that marks datapoints as stale."
    )]
    #[instrument(skip_all)]
    async fn delete_datapoints(
        &self,
        Parameters(params): Parameters<DeleteDatapointsToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let request = DeleteDatapointsRequest { ids: params.ids };
        let response = match delete_datapoints(&database, &params.dataset_name, request).await {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Create datapoints from existing inferences. Can specify inference IDs directly or use a query to find inferences. Supports choosing the output source (inference output or demonstration feedback)."
    )]
    #[instrument(skip_all)]
    async fn create_datapoints_from_inferences(
        &self,
        Parameters(params): Parameters<CreateDatapointsFromInferencesToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let request = CreateDatapointsFromInferenceRequest {
            params: params.params,
        };
        let response = match create_from_inferences(
            &self.app_state.config,
            &database,
            params.dataset_name,
            request,
        )
        .await
        {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Run a non-streaming inference using a TensorZero function or model. Returns the model response including content, usage, and metadata."
    )]
    #[instrument(skip_all)]
    async fn inference(
        &self,
        Parameters(params): Parameters<InferenceToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let app_state = &self.app_state;
        let inference_params = params.into_params(Default::default());
        let output = match inference(
            app_state.config.clone(),
            &app_state.http_client,
            app_state.clickhouse_connection_info.clone(),
            app_state.postgres_connection_info.clone(),
            app_state.cache_manager.clone(),
            app_state.deferred_tasks.clone(),
            app_state.rate_limiting_manager.clone(),
            app_state.primary_datastore,
            inference_params,
            None,
        )
        .await
        {
            Ok(output) => output,
            Err(e) => return handle_tool_error(e),
        };

        let response = match output.output {
            InferenceOutput::NonStreaming(response) => response,
            InferenceOutput::Streaming(_) => {
                return Err(McpError::internal_error(
                    "Unexpected streaming response from non-streaming inference".to_string(),
                    None,
                ));
            }
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Submit feedback for an inference or episode. Supports comments, demonstrations, and float/boolean metric values."
    )]
    #[instrument(skip_all)]
    async fn feedback(
        &self,
        Parameters(params): Parameters<FeedbackToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let feedback_params = params.into_params(Default::default());
        let response = match feedback(self.app_state.as_ref().clone(), feedback_params, None).await
        {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Get feedback statistics (mean, variance, count) by variant for a function and metric. Useful for analyzing variant performance."
    )]
    #[instrument(skip_all)]
    async fn get_feedback_by_variant(
        &self,
        Parameters(params): Parameters<GetFeedbackByVariantToolParams>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let response = match get_feedback_by_variant(
            &database,
            &params.metric_name,
            &params.function_name,
            params.variant_names.as_ref(),
        )
        .await
        {
            Ok(response) => response,
            Err(e) => return handle_tool_error(e),
        };

        let json = serde_json::to_string(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[tool_handler]
impl ServerHandler for TensorZeroMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("TensorZero", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "TensorZero MCP Server - query observability data from TensorZero".to_string(),
            )
    }
}
