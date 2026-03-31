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

/// Create a datapoint in the given dataset and return the datapoint ID.
pub async fn create_test_datapoint(dataset_name: &str) -> String {
    let client = reqwest::Client::new();
    let response = client
        .post(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&json!({
            "datapoints": [{
                "type": "chat",
                "function_name": "basic_test",
                "input": {
                    "system": {"assistant_name": "TestBot"},
                    "messages": [{"role": "user", "content": "Hello"}]
                },
                "output": [{"type": "text", "text": "Hi there!"}]
            }]
        }))
        .send()
        .await
        .unwrap();
    assert!(
        response.status().is_success(),
        "Failed to create datapoint: {:?}",
        response.text().await
    );
    let body: Value = response.json().await.unwrap();
    body["ids"][0].as_str().unwrap().to_string()
}

/// Submit boolean feedback for an inference and return the feedback ID.
pub async fn submit_boolean_feedback(inference_id: &str, metric_name: &str, value: bool) -> String {
    let client = reqwest::Client::new();
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&json!({
            "inference_id": inference_id,
            "metric_name": metric_name,
            "value": value,
        }))
        .send()
        .await
        .unwrap();
    assert!(
        response.status().is_success(),
        "Failed to submit feedback: {:?}",
        response.text().await
    );
    let body: Value = response.json().await.unwrap();
    body["feedback_id"].as_str().unwrap().to_string()
}

/// Poll an MCP tool until the given condition is met on the response.
/// Returns the first response that satisfies the condition.
/// Panics if the condition is not met within the timeout.
pub async fn poll_mcp_tool<F>(
    mcp: &McpTestClient,
    tool_name: &str,
    params: Value,
    condition: F,
) -> Value
where
    F: Fn(&Value) -> bool,
{
    let max_attempts = 10;
    let delay = std::time::Duration::from_millis(500);
    for _ in 0..max_attempts {
        let response: Value = mcp.call_tool(tool_name, params.clone()).await;
        if condition(&response) {
            return response;
        }
        tokio::time::sleep(delay).await;
    }
    panic!(
        "Condition not met after {max_attempts} attempts for tool `{tool_name}` with params {params}"
    );
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
