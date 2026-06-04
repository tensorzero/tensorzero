use rmcp::{RoleClient, ServiceExt, model::CallToolResult, service::RunningService};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tensorzero::{
    ClientInferenceParams, CreateChatDatapointRequest, CreateDatapointRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsRequest, FeedbackParams, Input,
    InputMessage, InputMessageContent, Role, System,
};
use tensorzero_core::inference::types::{Arguments, ContentBlockChatOutput, Text};

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
    pub async fn call_tool<T: DeserializeOwned, P: Serialize>(&self, name: &str, params: P) -> T {
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
    pub async fn call_tool_raw<P: Serialize>(&self, name: &str, params: P) -> CallToolResult {
        let params = serde_json::to_value(params).expect("tool params should serialize");
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

#[derive(Serialize)]
pub struct DatasetToolParams<T: Serialize> {
    pub dataset_name: String,
    #[serde(flatten)]
    pub request: T,
}

#[derive(Serialize)]
pub struct OptionalDatasetToolParams<T: Serialize> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_name: Option<String>,
    #[serde(flatten)]
    pub request: T,
}

#[derive(Serialize)]
pub struct InferenceToolParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_name: Option<String>,
    pub input: Input,
}

#[derive(Serialize)]
pub struct FeedbackByTargetIdToolParams {
    pub target_id: uuid::Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

#[derive(Serialize)]
pub struct LatestFeedbackByMetricToolParams {
    pub target_id: uuid::Uuid,
}

#[derive(Serialize)]
pub struct FeedbackByVariantToolParams {
    pub metric_name: String,
    pub function_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_names: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct ListEpisodesToolParams {
    pub limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_name: Option<String>,
}

#[derive(Serialize)]
pub struct CreateDatapointsFromInferencesToolParams {
    pub dataset_name: String,
    pub params: CreateDatapointsFromInferenceRequestParams,
}

pub fn chat_input(message: &str) -> Input {
    Input {
        system: Some(System::Template(Arguments(
            [("assistant_name".to_string(), json!("TestBot"))]
                .into_iter()
                .collect(),
        ))),
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![InputMessageContent::Text(Text {
                text: message.to_string(),
            })],
        }],
    }
}

pub fn chat_input_without_system(message: &str) -> Input {
    Input {
        system: None,
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![InputMessageContent::Text(Text {
                text: message.to_string(),
            })],
        }],
    }
}

pub fn chat_output(text: &str) -> Vec<ContentBlockChatOutput> {
    vec![ContentBlockChatOutput::Text(Text {
        text: text.to_string(),
    })]
}

/// Create a datapoint in the given dataset and return the datapoint ID.
pub async fn create_test_datapoint(dataset_name: &str) -> String {
    let client = reqwest::Client::new();
    let request = CreateDatapointsRequest {
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "basic_test".to_string(),
            episode_id: None,
            input: chat_input("Hello"),
            output: Some(chat_output("Hi there!")),
            dynamic_tool_params: Default::default(),
            tags: None,
            name: None,
        })],
    };
    let response = client
        .post(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints"
        )))
        .json(&request)
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
    let request = FeedbackParams {
        inference_id: Some(inference_id.parse().expect("inference ID should be a UUID")),
        metric_name: metric_name.to_string(),
        value: json!(value),
        ..Default::default()
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&request)
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
    let mut last_response = Value::Null;
    for _ in 0..max_attempts {
        let response: Value = mcp.call_tool(tool_name, params.clone()).await;
        if condition(&response) {
            return response;
        }
        last_response = response;
        tokio::time::sleep(delay).await;
    }
    panic!(
        "Condition not met after {max_attempts} attempts for tool `{tool_name}` with params {params}\nLast response: {last_response}"
    );
}

/// Insert an inference and return (inference_id, episode_id).
pub async fn insert_inference(function_name: &str) -> (String, String) {
    let client = reqwest::Client::new();
    let request = ClientInferenceParams {
        function_name: Some(function_name.to_string()),
        input: chat_input("Hello"),
        stream: Some(false),
        ..Default::default()
    };
    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&request)
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
