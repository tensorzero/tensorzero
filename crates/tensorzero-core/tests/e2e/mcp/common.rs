use rmcp::{RoleClient, ServiceExt, model::CallToolResult, service::RunningService};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::common::get_gateway_endpoint;

pub struct McpTestClient {
    client: RunningService<RoleClient, ()>,
}

impl McpTestClient {
    pub async fn connect() -> Self {
        let mcp_url = get_gateway_endpoint("/mcp").to_string();
        let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(mcp_url.as_str());
        let client = ().serve(transport).await.expect("Failed to connect MCP client");
        Self { client }
    }

    pub async fn call_tool<T: DeserializeOwned>(&self, name: &str, params: Value) -> T {
        let result = self.call_tool_raw(name, params).await;
        assert!(
            !result.is_error.unwrap_or(false),
            "MCP tool `{name}` returned an error: {result:?}"
        );
        let text = result
            .content
            .iter()
            .find_map(|c| c.as_text())
            .expect("Expected text content in CallToolResult")
            .text
            .as_str();
        serde_json::from_str(text).expect("Failed to deserialize MCP tool response")
    }

    pub async fn call_tool_raw(&self, name: &str, params: Value) -> CallToolResult {
        let args = params
            .as_object()
            .expect("params must be a JSON object")
            .clone();
        let mut params = rmcp::model::CallToolRequestParams::default();
        params.name = name.to_string().into();
        params.arguments = Some(args);
        self.client
            .call_tool(params)
            .await
            .expect("MCP call_tool RPC failed")
    }

    pub async fn list_tools(&self) -> Vec<rmcp::model::Tool> {
        self.client
            .list_tools(Default::default())
            .await
            .expect("list_tools failed")
            .tools
    }

    pub async fn cancel(self) {
        let _ = self.client.cancel().await;
    }
}
