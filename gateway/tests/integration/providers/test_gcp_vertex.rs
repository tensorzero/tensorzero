use crate::integration::providers::common::create_simple_inference_request;
use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::{inference::providers::gcp_vertex::GCPVertexGeminiProvider, model::ProviderConfig};
use serde_json::{json, Value};

#[tokio::test]
async fn test_infer() {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    let config: ProviderConfig = serde_json::from_value(provider_config_json)
        .expect("Failed to deserialize provider config");
    let client = reqwest::Client::new();
    let inference_request = create_simple_inference_request();
    let result = GCPVertexGeminiProvider::infer(&inference_request, &config, &client).await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
    assert!(result.unwrap().content.is_some());
}

// TODO: add tests for streaming and tool calls
// #[tokio::test]
// async fn test_infer_stream() {
// }

// #[tokio::test]
// async fn test_infer_with_tool_calls() {
// }
