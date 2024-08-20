use futures::StreamExt;
use serde_json::{json, Value};

use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::types::{ContentBlock, Text};
use gateway::model::ProviderConfig;

use crate::providers::common::{
    create_json_inference_request, create_streaming_json_inference_request,
    test_simple_inference_request_with_provider, test_streaming_inference_request_with_provider,
};

#[tokio::test]
async fn test_simple_inference_request() {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();

    test_simple_inference_request_with_provider(provider).await;
}

#[tokio::test]
async fn test_streaming_inference_request() {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();

    test_streaming_inference_request_with_provider(provider).await;
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

// #[tokio::test]
// async fn test_infer_with_tool_calls() {
// }

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
    // Third as an arbitrary middle chunk, the first and last may contain only metadata
    assert!(collected_chunks[3].content.len() == 1);
    assert!(collected_chunks.last().unwrap().usage.is_some());
}
