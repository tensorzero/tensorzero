use futures::StreamExt;
use serde_json::{json, Value};

use gateway::inference::providers::gcp_vertex::GCPVertexGeminiProvider;
use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::types::{ContentBlock, ContentBlockChunk, Text};
use gateway::model::ProviderConfig;

use crate::providers::common::{
    create_json_inference_request, create_streaming_json_inference_request,
    create_streaming_tool_inference_request, create_tool_use_inference_request,
    TestableProviderConfig,
};

crate::enforce_provider_tests!(GCPVertexGeminiProvider);

impl TestableProviderConfig for GCPVertexGeminiProvider {
    async fn get_simple_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_streaming_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_tool_use_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }
}

/// Get a generic provider for testing
fn get_provider() -> ProviderConfig {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    serde_json::from_value(provider_config_json).unwrap()
}

// Gemini Flash does not support JSON mode using an output schema -- the model provider knows this automatically
// We test the Flash and Pro here so that we can test both code paths
#[tokio::test]
async fn test_json_request_flash() {
    // Load API key from environment variable
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();
    let client = reqwest::Client::new();
    let inference_request = create_json_inference_request();

    let result = provider.infer(&inference_request, &client).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.content.len() == 1);
    let content = result.content.first().unwrap();
    match content {
        ContentBlock::Text(Text { text }) => {
            // parse the result text and see if it matches the output schema
            let result_json: serde_json::Value = serde_json::from_str(text)
                .map_err(|_| format!(r#"Failed to parse JSON: "{text}""#))
                .unwrap();
            assert!(result_json.get("honest_answer").is_some());
            assert!(result_json.get("mischevious_answer").is_some());
        }
        _ => panic!("Expected a text content block"),
    }
}

/// Test Gemini Pro JSON mode with an output schema
#[tokio::test]
async fn test_json_request_pro() {
    // Load API key from environment variable
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-pro-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();
    let client = reqwest::Client::new();
    let inference_request = create_json_inference_request();

    let result = provider.infer(&inference_request, &client).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.content.len() == 1);
    let content = result.content.first().unwrap();
    match content {
        ContentBlock::Text(Text { text }) => {
            // parse the result text and see if it matches the output schema
            let result_json: serde_json::Value = serde_json::from_str(text)
                .map_err(|_| format!(r#"Failed to parse JSON: "{text}""#))
                .unwrap();
            assert!(result_json.get("honest_answer").is_some());
            assert!(result_json.get("mischevious_answer").is_some());
        }
        _ => panic!("Expected a text content block"),
    }
}

#[tokio::test]
async fn test_infer_with_tool_calls_pro() {
    // Load API key from environment variable
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-pro-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();
    let client = reqwest::Client::new();

    let inference_request = create_tool_use_inference_request();

    let result = provider.infer(&inference_request, &client).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.content.len() == 1);
    let content = response.content.first().unwrap();
    match content {
        ContentBlock::ToolCall(tool_call) => {
            assert!(tool_call.name == "get_weather");
            let arguments: serde_json::Value = serde_json::from_str(&tool_call.arguments)
                .expect("Failed to parse tool call arguments");
            assert!(arguments.get("location").is_some());
        }
        _ => panic!("Expected a tool call"),
    }
}

/// Gemini Flash does not support tool calls with Any but we test to see if the Any mode is handled correctly (best effort with Auto)
#[tokio::test]
async fn test_infer_with_tool_calls_flash() {
    // Load API key from environment variable
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();
    let client = reqwest::Client::new();

    let inference_request = create_tool_use_inference_request();

    let result = provider.infer(&inference_request, &client).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.content.len() == 1);
    let content = response.content.first().unwrap();
    match content {
        ContentBlock::ToolCall(tool_call) => {
            assert!(tool_call.name == "get_weather");
            let arguments: serde_json::Value = serde_json::from_str(&tool_call.arguments)
                .expect("Failed to parse tool call arguments");
            assert!(arguments.get("location").is_some());
        }
        _ => panic!("Expected a tool call"),
    }
}

#[tokio::test]
async fn test_infer_stream_with_tool_calls() {
    // Load API key from environment variable
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();
    let client = reqwest::Client::new();

    let inference_request = create_streaming_tool_inference_request();

    let result = provider.infer_stream(&inference_request, &client).await;
    assert!(result.is_ok());
    let (chunk, mut stream) = result.unwrap();
    let mut collected_chunks = vec![chunk];
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok());
        collected_chunks.push(chunk.unwrap());
    }
    assert!(!collected_chunks.is_empty());
    // As far as I can tell GCP does not stream tool calls with more than 1 chunk
    assert!(collected_chunks[0].content.len() == 1);
    assert!(collected_chunks.last().unwrap().usage.is_some());
    let first_chunk = collected_chunks.first().unwrap();
    match first_chunk.content.first().unwrap() {
        ContentBlockChunk::ToolCall(tool_call) => {
            assert!(tool_call.name == "get_weather");
        }
        _ => panic!("Expected a tool call"),
    }
}

// Gemini Flash does not support JSON mode using an output schema -- the model provider knows this automatically
// We test the Flash and Pro here so that we can test both code paths
#[tokio::test]
async fn test_streaming_json_request_flash() {
    // Load API key from environment variable
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();
    let client = reqwest::Client::new();
    let inference_request = create_streaming_json_inference_request();

    let result = provider.infer_stream(&inference_request, &client).await;
    assert!(result.is_ok());
    let (chunk, mut stream) = result.unwrap();
    let mut collected_chunks = vec![chunk];
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok());
        collected_chunks.push(chunk.unwrap());
    }
    assert!(!collected_chunks.is_empty());
    // 1th as an arbitrary middle chunk, the first and last may contain only metadata
    assert!(collected_chunks[1].content.len() == 1);
    assert!(collected_chunks.last().unwrap().usage.is_some());
}
