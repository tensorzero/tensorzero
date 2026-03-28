use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use tensorzero_core::error::Error;
use tracing::instrument;

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
