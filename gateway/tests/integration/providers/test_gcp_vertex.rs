use futures::StreamExt;
use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::types::{ContentBlock, Text};
use gateway::model::ProviderConfig;
use serde_json::{json, Value};

use crate::integration::providers::common::{
    create_json_inference_request, create_simple_inference_request,
    create_streaming_inference_request,
};

#[tokio::test]
async fn test_infer() {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();
    let client = reqwest::Client::new();
    let inference_request = create_simple_inference_request();
    let result = provider.infer(&inference_request, &client).await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let result = result.unwrap();
    assert!(result.content.len() == 1);
    let content = result.content.first().unwrap();
    match content {
        ContentBlock::Text(Text { text }) => {
            assert!(!text.is_empty());
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn test_infer_stream() {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let provider: ProviderConfig = serde_json::from_value(provider_config_json).unwrap();
    let client = reqwest::Client::new();
    let inference_request = create_streaming_inference_request();
    let result = provider.infer_stream(&inference_request, &client).await;
    let (chunk, mut stream) = result.unwrap();
    let mut collected_chunks = vec![chunk];
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok(), "{}", chunk.unwrap_err());
        collected_chunks.push(chunk.unwrap());
    }
    assert!(!collected_chunks.is_empty());
    // Fourth as an arbitrary middle chunk, the first and last may contain only metadata
    assert!(collected_chunks[4].content.len() == 1);
    assert!(collected_chunks.last().unwrap().usage.is_some());
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
        _ => unreachable!(),
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
    println!("{:?}", result);
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
        _ => unreachable!(),
    }
}

// #[tokio::test]
// async fn test_infer_with_tool_calls() {
// }
