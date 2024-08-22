use serde_json::{json, Value};

use gateway::model::ProviderConfig;

use crate::providers::common::IntegrationTestProviders;

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> IntegrationTestProviders {
    IntegrationTestProviders::with_providers(vec![
        get_provider_gemini_flash(),
        get_provider_gemini_pro(),
    ])
}

/// Get a Gemini Flash provider for testing
fn get_provider_gemini_flash() -> ProviderConfig {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-flash-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    serde_json::from_value(provider_config_json).unwrap()
}

/// Get a Gemini Pro provider for testing
fn get_provider_gemini_pro() -> ProviderConfig {
    let mut provider_config_json = json!({"type": "gcp_vertex_gemini", "model_id": "gemini-1.5-pro-001", "location": "us-central1"});
    let gcp_project_id = "tensorzero-public";
    provider_config_json["project_id"] = Value::String(gcp_project_id.to_string());
    serde_json::from_value(provider_config_json).unwrap()
}
