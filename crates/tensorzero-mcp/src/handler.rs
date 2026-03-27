use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};

use tensorzero_core::endpoints::stored_inferences::v1::types::{
    GetInferencesRequest, ListInferencesRequest,
};
use tensorzero_core::endpoints::stored_inferences::v1::{get_inferences, list_inferences};
use tensorzero_core::utils::gateway::AppStateData;

#[derive(Clone)]
pub struct TensorZeroMcpServer {
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
    async fn list_inferences(
        &self,
        Parameters(request): Parameters<ListInferencesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let response = list_inferences(&self.app_state.config, &database, request)
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to list inferences: {e}"), None)
            })?;

        let json = serde_json::to_string_pretty(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Retrieve specific inferences by their IDs. Returns full inference data including inputs, outputs, function/variant names, timestamps, and tags."
    )]
    async fn get_inferences(
        &self,
        Parameters(request): Parameters<GetInferencesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let database = self.app_state.get_delegating_database();
        let response = get_inferences(&self.app_state.config, &database, request)
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to get inferences: {e}"), None)
            })?;

        let json = serde_json::to_string_pretty(&response).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize response: {e}"), None)
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[tool_handler]
impl ServerHandler for TensorZeroMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "TensorZero MCP Server - query observability data from TensorZero".to_string(),
            )
    }
}
