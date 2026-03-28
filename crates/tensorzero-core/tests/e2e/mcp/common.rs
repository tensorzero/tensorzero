use rmcp::{RoleClient, ServiceExt, model::CallToolResult, service::RunningService};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

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

    /// Call an MCP tool, assert it succeeded, and deserialize the response.
    /// Panics on error responses — use `call_tool_raw` for tests that expect failures.
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

    /// Call an MCP tool and return the raw `CallToolResult` without assertions or deserialization.
    /// Use this for tests that need to inspect error responses.
    pub async fn call_tool_raw(&self, name: &str, params: Value) -> CallToolResult {
        let args = params
            .as_object()
            .expect("params must be a JSON object")
            .clone();
        let mut params = rmcp::model::CallToolRequestParams::default();
        params.name = name.to_string().into();
        params.arguments = Some(args);
        match self.client.call_tool(params).await {
            Ok(result) => result,
            // JSON-RPC errors (e.g. deserialization failures) are treated as tool errors
            Err(e) => CallToolResult::error(vec![rmcp::model::Content::text(format!("{e:?}"))]),
        }
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

/// Insert an inference and return (inference_id, episode_id).
pub async fn insert_inference(function_name: &str) -> (String, String) {
    let client = reqwest::Client::new();
    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&json!({
            "function_name": function_name,
            "input": {
                "system": {"assistant_name": "TestBot"},
                "messages": [{"role": "user", "content": "Hello"}]
            },
            "stream": false,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        response.status().is_success(),
        "Inference request failed: {:?}",
        response.status()
    );
    let body: Value = response.json().await.unwrap();
    let inference_id = body["inference_id"].as_str().unwrap().to_string();
    let episode_id = body["episode_id"].as_str().unwrap().to_string();
    (inference_id, episode_id)
}
