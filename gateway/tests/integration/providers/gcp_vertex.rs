use serde_json::{json, Value};

use gateway::inference::providers::gcp_vertex::GCPVertexGeminiProvider;
use gateway::model::ProviderConfig;

use crate::providers::common::TestableProviderConfig;

crate::generate_provider_tests!(GCPVertexGeminiProvider);

impl TestableProviderConfig for GCPVertexGeminiProvider {
    async fn get_simple_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider_gemini_flash(), get_provider_gemini_pro()]
    }

    async fn get_streaming_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider_gemini_flash(), get_provider_gemini_pro()]
    }

    async fn get_tool_use_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider_gemini_flash(), get_provider_gemini_pro()]
    }

    async fn get_tool_use_streaming_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider_gemini_flash(), get_provider_gemini_pro()]
    }

    async fn get_json_mode_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider_gemini_flash(), get_provider_gemini_pro()]
    }

    async fn get_json_mode_streaming_inference_request_providers() -> Vec<ProviderConfig> {
        vec![get_provider_gemini_flash(), get_provider_gemini_pro()]
    }
}

/// Get a generic provider for testing
fn get_provider_gemini_flash() -> ProviderConfig {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    serde_json::from_value(provider_config_json).unwrap()
}

/// Get a generic provider for testing
fn get_provider_gemini_pro() -> ProviderConfig {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-pro-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    serde_json::from_value(provider_config_json).unwrap()
}
